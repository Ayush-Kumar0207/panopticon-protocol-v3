# Transfer a Google Drive folder tree between personal Gmail accounts

This notebook flow transfers ownership from `krayush70049@gmail.com` to
`kumarayush70049@gmail.com`, including every source-owned file and subfolder
under the selected folder.

Important facts:

- Transferring only a folder does **not** transfer ownership of its contents.
- Personal Gmail-to-Gmail transfers require two phases: the old owner requests
  each transfer, then the new owner accepts each transfer.
- Files owned by somebody else are audited but skipped. They do not consume the
  old account's quota.
- A shortcut and the containing folder do not determine storage usage. Files
  created/uploaded while authenticated as the source account normally remain
  owned by, and charged to, the source account even inside a target-owned
  shared folder.
- Shared-drive items cannot have individual ownership transferred.
- The screenshot's `panopticon-fixed-v3-ep20` entry appears to be a shortcut.
  The audit cell resolves a root shortcut to its real target folder.
- Do not add, remove, or move files in the selected tree while running this.

The process is resumable and idempotent. Re-run a phase after an interruption;
items already pending or already transferred are skipped.

## Manual setup

1. Open the selected folder or shortcut in Drive and copy its URL.
2. Open a new Google Colab notebook while signed in to
   `krayush70049@gmail.com`.
3. Run Cells 1-6 for the request phase and download the generated manifest.
4. Open a **fresh Incognito window or separate Chrome profile**, signed in only
   to `kumarayush70049@gmail.com`, and open a fresh Colab notebook.
5. In the fresh notebook, run Cells 1-2, then Cells 7-9. Upload the manifest
   downloaded in step 3 when prompted.

## Cell 1 - Authenticate and show the active account

```python
from google.colab import auth, files as colab_files
auth.authenticate_user()

import json
import random
import re
import time
from collections import deque
from datetime import datetime, timezone

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

credentials, _ = google.auth.default()
drive = build("drive", "v3", credentials=credentials, cache_discovery=False)

about = drive.about().get(
    fields="user(emailAddress,permissionId),storageQuota"
).execute()
print(json.dumps(about, indent=2))
```

Check the printed `user.emailAddress` before continuing.

## Cell 2 - Shared helper functions

```python
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
MANIFEST_PATH = "/content/drive_ownership_transfer_manifest.json"

RETRYABLE_REASONS = {
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "backendError",
    "internalError",
}

def api(call, attempts=9):
    """Execute a newly-created API request with bounded exponential backoff."""
    for attempt in range(attempts):
        try:
            return call()
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            reason = ""
            try:
                payload = json.loads(exc.content.decode("utf-8"))
                reason = payload["error"]["errors"][0].get("reason", "")
            except Exception:
                pass

            retryable = (
                status in {429, 500, 502, 503, 504}
                or reason in RETRYABLE_REASONS
            )
            if not retryable or attempt == attempts - 1:
                raise

            delay = min(64, 2 ** attempt) + random.random()
            print(f"Retryable Drive API error ({status}/{reason}); sleeping {delay:.1f}s")
            time.sleep(delay)

def error_text(exc):
    if isinstance(exc, HttpError):
        try:
            return exc.content.decode("utf-8")
        except Exception:
            pass
    return f"{type(exc).__name__}: {exc}"

def account_info():
    return api(lambda: drive.about().get(
        fields="user(emailAddress,permissionId),storageQuota"
    ).execute())

def assert_account(expected_email):
    info = account_info()
    actual = info["user"]["emailAddress"].lower()
    expected = expected_email.lower()
    if actual != expected:
        raise RuntimeError(
            f"Wrong Google account: authenticated as {actual}; expected {expected}"
        )
    print(f"Authenticated correctly as {actual}")
    return info

def parse_drive_id(value):
    value = value.strip()
    for pattern in (
        r"/folders/([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{15,}", value):
        return value
    raise ValueError("Could not extract a Google Drive ID from ROOT_URL_OR_ID")

ITEM_FIELDS = (
    "id,name,mimeType,parents,owners(emailAddress),ownedByMe,driveId,"
    "quotaBytesUsed,shortcutDetails(targetId,targetMimeType)"
)

def get_item(file_id):
    return api(lambda: drive.files().get(
        fileId=file_id,
        fields=ITEM_FIELDS,
        supportsAllDrives=True,
    ).execute())

def resolve_root_shortcut(file_id):
    seen = set()
    item = get_item(file_id)
    while item["mimeType"] == SHORTCUT_MIME:
        if item["id"] in seen:
            raise RuntimeError("Shortcut loop detected")
        seen.add(item["id"])
        target_id = item.get("shortcutDetails", {}).get("targetId")
        if not target_id:
            raise RuntimeError("Cannot resolve the shortcut target")
        print(f"Resolved shortcut {item['name']} -> {target_id}")
        item = get_item(target_id)
    return item

def list_children(folder_id):
    result = []
    token = None
    while True:
        response = api(lambda: drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields=f"nextPageToken,files({ITEM_FIELDS})",
            pageSize=100,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute())
        result.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return result

def normalized_record(item, depth, path):
    return {
        "id": item["id"],
        "name": item.get("name", ""),
        "path": path,
        "depth": depth,
        "mimeType": item.get("mimeType"),
        "parents": item.get("parents", []),
        "ownerEmails": [
            owner.get("emailAddress", "").lower()
            for owner in item.get("owners", [])
        ],
        "ownedByMe": bool(item.get("ownedByMe")),
        "driveId": item.get("driveId"),
        "quotaBytesUsed": int(item.get("quotaBytesUsed", "0") or 0),
        "shortcutTargetId": item.get("shortcutDetails", {}).get("targetId"),
    }

def inventory_tree(root_item):
    records = []
    listing_errors = []
    queue = deque([(root_item, 0, root_item["name"])])
    seen = {root_item["id"]}

    while queue:
        item, depth, path = queue.popleft()
        records.append(normalized_record(item, depth, path))

        if item["mimeType"] != FOLDER_MIME:
            continue

        try:
            children = list_children(item["id"])
        except Exception as exc:
            listing_errors.append({
                "folderId": item["id"],
                "path": path,
                "error": error_text(exc),
            })
            continue

        for child in children:
            if child["id"] in seen:
                continue
            seen.add(child["id"])
            queue.append((child, depth + 1, f"{path}/{child['name']}"))

    return records, listing_errors

def list_permissions(file_id):
    permissions = []
    token = None
    while True:
        response = api(lambda: drive.permissions().list(
            fileId=file_id,
            fields="nextPageToken,permissions(id,emailAddress,role,pendingOwner,type)",
            pageSize=100,
            pageToken=token,
            supportsAllDrives=True,
        ).execute())
        permissions.extend(response.get("permissions", []))
        token = response.get("nextPageToken")
        if not token:
            return permissions

def find_user_permission(file_id, email, permission_id=None):
    email = email.lower()
    for permission in list_permissions(file_id):
        if permission.get("type") != "user":
            continue
        if permission.get("emailAddress", "").lower() == email:
            return permission
        if permission_id and permission.get("id") == permission_id:
            return permission
    return None

def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
```

## Cell 3 - Source account configuration and dry-run audit

Run this cell while authenticated as `krayush70049@gmail.com`.

```python
SOURCE_EMAIL = "krayush70049@gmail.com"
TARGET_EMAIL = "kumarayush70049@gmail.com"

# Paste the selected folder/shortcut URL, or its Drive ID.
ROOT_URL_OR_ID = "PASTE_FOLDER_OR_SHORTCUT_URL_HERE"

assert_account(SOURCE_EMAIL)
input_id = parse_drive_id(ROOT_URL_OR_ID)
root = resolve_root_shortcut(input_id)

if root["mimeType"] != FOLDER_MIME:
    raise RuntimeError("The resolved root is not a folder")
if root.get("driveId"):
    raise RuntimeError("The root is in a Shared Drive; individual ownership cannot transfer")
root_owners = [
    owner.get("emailAddress", "").lower()
    for owner in root.get("owners", [])
]
if root.get("ownedByMe"):
    print("The source account owns the root; it will be transferred last.")
elif TARGET_EMAIL.lower() in root_owners:
    print("The target account already owns the root; only source-owned descendants will transfer.")
else:
    print(f"WARNING: the root is owned by another account: {root_owners}")

items, listing_errors = inventory_tree(root)
eligible = [item for item in items if item["ownedByMe"] and not item["driveId"]]
skipped = [item for item in items if item not in eligible]
estimated_bytes = sum(item["quotaBytesUsed"] for item in eligible)

manifest = {
    "version": 1,
    "createdAt": datetime.now(timezone.utc).isoformat(),
    "sourceEmail": SOURCE_EMAIL.lower(),
    "targetEmail": TARGET_EMAIL.lower(),
    "inputId": input_id,
    "rootId": root["id"],
    "rootName": root["name"],
    "items": items,
    "listingErrors": listing_errors,
}
save_manifest(manifest)

print(f"Resolved root: {root['name']} ({root['id']})")
print(f"Total accessible items: {len(items)}")
print(f"Source-owned items eligible for transfer: {len(eligible)}")
print(f"Skipped items owned elsewhere/shared-drive items: {len(skipped)}")
print(f"Estimated quota to move: {estimated_bytes / (1024**3):.3f} GiB")
print(f"Folder-listing errors: {len(listing_errors)}")

if listing_errors:
    raise RuntimeError("Audit is incomplete because some folders could not be listed")
```

Read the counts carefully. This is the dry run; it changes nothing.

## Cell 4 - Request pending ownership for every eligible item

This deliberately processes the deepest items first and the root folder last.

```python
assert_account(SOURCE_EMAIL)
confirmation = input(
    f"Type REQUEST TRANSFER to request {len(eligible)} ownership transfers: "
)
if confirmation != "REQUEST TRANSFER":
    raise RuntimeError("Cancelled")

ordered = sorted(eligible, key=lambda item: (-item["depth"], item["path"]))

def request_pending_owner(item):
    permission = find_user_permission(item["id"], TARGET_EMAIL)
    if permission and permission.get("role") == "owner":
        return "already_target_owner"
    if permission and permission.get("pendingOwner"):
        return "already_pending"

    body = {"role": "writer", "pendingOwner": True}
    if permission:
        try:
            api(lambda: drive.permissions().update(
                fileId=item["id"],
                permissionId=permission["id"],
                body=body,
                supportsAllDrives=True,
                fields="id,emailAddress,role,pendingOwner",
            ).execute())
            return "pending_updated"
        except HttpError:
            # A propagated folder permission can be non-updatable on a child.
            # Creating a direct pending-owner permission is the fallback.
            pass

    api(lambda: drive.permissions().create(
        fileId=item["id"],
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": TARGET_EMAIL,
            "pendingOwner": True,
        },
        sendNotificationEmail=True,
        supportsAllDrives=True,
        fields="id,emailAddress,role,pendingOwner",
    ).execute())
    return "pending_created"

phase1_errors = []
for index, item in enumerate(ordered, start=1):
    try:
        status = request_pending_owner(item)
        item["phase1"] = {"status": status}
        print(f"[{index}/{len(ordered)}] {status}: {item['path']}")
    except Exception as exc:
        item["phase1"] = {"status": "error", "error": error_text(exc)}
        phase1_errors.append(item)
        print(f"[{index}/{len(ordered)}] ERROR: {item['path']}")

    if index % 20 == 0:
        save_manifest(manifest)
    time.sleep(0.12)

manifest["phase1CompletedAt"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print(f"Phase 1 errors: {len(phase1_errors)}")
```

If errors are non-zero, re-run Cell 4. Existing pending transfers are skipped.

## Cell 5 - Verify all pending requests

```python
assert_account(SOURCE_EMAIL)
pending_failures = []

for index, item in enumerate(ordered, start=1):
    permission = find_user_permission(item["id"], TARGET_EMAIL)
    good = bool(
        permission
        and (
            permission.get("pendingOwner")
            or permission.get("role") == "owner"
        )
    )
    if not good:
        pending_failures.append(item)
        print(f"NOT PENDING: {item['path']}")
    if index % 50 == 0:
        print(f"Verified {index}/{len(ordered)}")

print(f"Pending verification failures: {len(pending_failures)}")
if pending_failures:
    raise RuntimeError("Re-run Cell 4 before switching accounts")
```

## Cell 6 - Download the cross-account manifest

```python
save_manifest(manifest)
colab_files.download(MANIFEST_PATH)
```

Now use a fresh Incognito window or separate browser profile signed in only to
`kumarayush70049@gmail.com`. In a fresh Colab notebook, run Cells 1-2, then
continue below.

## Cell 7 - Upload manifest and validate target account/quota

```python
uploaded = colab_files.upload()
manifest_name = next(name for name in uploaded if name.endswith(".json"))
manifest = json.loads(uploaded[manifest_name].decode("utf-8"))

SOURCE_EMAIL = manifest["sourceEmail"]
TARGET_EMAIL = manifest["targetEmail"]
ROOT_ID = manifest["rootId"]
eligible = [
    item for item in manifest["items"]
    if item["ownedByMe"] and not item["driveId"]
]

target_info = assert_account(TARGET_EMAIL)
target_permission_id = target_info["user"]["permissionId"]
estimated_bytes = sum(item["quotaBytesUsed"] for item in eligible)
quota = target_info.get("storageQuota", {})

limit = int(quota["limit"]) if quota.get("limit") else None
usage = int(quota.get("usage", "0"))
free = (limit - usage) if limit is not None else None

print(f"Items to accept: {len(eligible)}")
print(f"Estimated quota to move: {estimated_bytes / (1024**3):.3f} GiB")
if free is not None:
    print(f"Target free quota: {free / (1024**3):.3f} GiB")
    if estimated_bytes > free:
        raise RuntimeError("Target account likely lacks enough free storage")
```

## Cell 8 - Accept transfers, repair hierarchy, and move root to My Drive

If the source owns the root, it is accepted only after all descendants succeed.
If the target already owns the root, the root is left untouched. If this cell
is interrupted or reports errors, run it again.

```python
assert_account(TARGET_EMAIL)
confirmation = input(
    f"Type ACCEPT TRANSFER to accept ownership of {len(eligible)} items: "
)
if confirmation != "ACCEPT TRANSFER":
    raise RuntimeError("Cancelled")

def accept_item(item):
    permission = find_user_permission(
        item["id"], TARGET_EMAIL, target_permission_id
    )
    if permission and permission.get("role") == "owner":
        return "already_owner"
    if not permission or not permission.get("pendingOwner"):
        raise RuntimeError("No pending-owner permission found")

    api(lambda: drive.permissions().update(
        fileId=item["id"],
        permissionId=permission["id"],
        body={"role": "owner"},
        transferOwnership=True,
        supportsAllDrives=True,
        fields="id,emailAddress,role,pendingOwner",
    ).execute())
    return "accepted"

root_item = next((item for item in eligible if item["id"] == ROOT_ID), None)
descendants = sorted(
    [item for item in eligible if not root_item or item["id"] != ROOT_ID],
    key=lambda item: (-item["depth"], item["path"]),
)

phase2_errors = []
for index, item in enumerate(descendants, start=1):
    try:
        status = accept_item(item)
        item["phase2"] = {"status": status}
        print(f"[{index}/{len(descendants)}] {status}: {item['path']}")
    except Exception as exc:
        item["phase2"] = {"status": "error", "error": error_text(exc)}
        phase2_errors.append(item)
        print(f"[{index}/{len(descendants)}] ERROR: {item['path']}")
    time.sleep(0.12)

if phase2_errors:
    save_manifest(manifest)
    root_note = " Root was NOT accepted." if root_item else ""
    raise RuntimeError(
        f"{len(phase2_errors)} items failed.{root_note} Re-run this cell."
    )

if root_item:
    root_item["phase2"] = {"status": accept_item(root_item)}
    print(f"Root: {root_item['phase2']['status']}: {root_item['path']}")
else:
    print("Root is not source-owned and was left untouched.")

# Repair any parent relationships changed during transfer.
eligible_ids = {item["id"] for item in eligible}
target_owned_container_ids = {
    item["id"]
    for item in manifest["items"]
    if TARGET_EMAIL.lower() in item.get("ownerEmails", [])
}
valid_repair_parent_ids = eligible_ids | target_owned_container_ids
repair_errors = []
for item in sorted(descendants, key=lambda item: (item["depth"], item["path"])):
    desired_parent = next(
        (parent for parent in item["parents"] if parent in valid_repair_parent_ids),
        None,
    )
    if not desired_parent:
        continue
    try:
        current = get_item(item["id"])
        current_parents = current.get("parents", [])
        if desired_parent not in current_parents:
            kwargs = {
                "fileId": item["id"],
                "addParents": desired_parent,
                "supportsAllDrives": True,
                "fields": "id,parents",
            }
            remove = [parent for parent in current_parents if parent != desired_parent]
            if remove:
                kwargs["removeParents"] = ",".join(remove)
            api(lambda kwargs=kwargs: drive.files().update(**kwargs).execute())
            print(f"Repaired parent: {item['path']}")
    except Exception as exc:
        repair_errors.append({"path": item["path"], "error": error_text(exc)})

# If the source owned the root, ensure it is visible in the target My Drive.
# An already target-owned root is intentionally left where it is.
if root_item:
    target_my_drive_root = api(
        lambda: drive.files().get(fileId="root", fields="id").execute()
    )["id"]
    current_root = get_item(ROOT_ID)
    current_root_parents = current_root.get("parents", [])
    if target_my_drive_root not in current_root_parents:
        kwargs = {
            "fileId": ROOT_ID,
            "addParents": target_my_drive_root,
            "supportsAllDrives": True,
            "fields": "id,parents",
        }
        remove = [p for p in current_root_parents if p != target_my_drive_root]
        if remove:
            kwargs["removeParents"] = ",".join(remove)
        api(lambda: drive.files().update(**kwargs).execute())
        print("Moved transferred root folder into target My Drive")

manifest["repairErrors"] = repair_errors
manifest["phase2CompletedAt"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print(f"Hierarchy repair errors: {len(repair_errors)}")
```

## Cell 9 - Final ownership verification and report download

```python
assert_account(TARGET_EMAIL)
verification_failures = []

for index, item in enumerate(eligible, start=1):
    try:
        current = get_item(item["id"])
        if not current.get("ownedByMe"):
            verification_failures.append({
                "path": item["path"],
                "reason": "target account is not owner",
            })
            print(f"NOT OWNED BY TARGET: {item['path']}")
    except Exception as exc:
        verification_failures.append({
            "path": item["path"],
            "reason": error_text(exc),
        })
    if index % 50 == 0:
        print(f"Verified {index}/{len(eligible)}")

manifest["verificationFailures"] = verification_failures
manifest["verifiedAt"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)

print(f"Final ownership failures: {len(verification_failures)}")
if verification_failures:
    raise RuntimeError("Re-run Cell 8, then Cell 9")

colab_files.download(MANIFEST_PATH)
print("Ownership transfer verified successfully.")
```

## Final manual checks

1. In `kumarayush70049@gmail.com`, confirm the transferred root appears in My
   Drive and opens normally.
2. In `krayush70049@gmail.com`, check Drive storage after Google finishes
   updating quota accounting. Ownership changes can take time to appear.
3. For a strongest final audit, authenticate a fresh Colab runtime as
   `krayush70049@gmail.com`, run Cells 1-3 using the same root ID, and confirm
   `Source-owned items eligible for transfer: 0`.
4. Do not remove the old account's Editor access until the hierarchy and files
   have been checked.
5. If API acceptance is blocked for a small number of items, sign in to the
   target account, search Drive for `pendingowner:me`, and accept those items
   manually. Then re-run Cell 9.

## Official references

- [Transfer file ownership - Drive API](https://developers.google.com/workspace/drive/api/guides/transfer-file)
- [Make someone else the owner of your file - Google Drive Help](https://support.google.com/drive/answer/2494892)
- [Drive API permissions.update](https://developers.google.com/workspace/drive/api/reference/rest/v3/permissions/update)
