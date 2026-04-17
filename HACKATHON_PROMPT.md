# Meta PyTorch OpenEnv Hackathon - Ideation Prompt

> **Instructions for use on April 25th:**
> Copy the following prompt, paste your newly revealed problem statement into the `[INSERT_YOUR_PROBLEM_STATEMENT_HERE]` placeholder, and send it to me (or any AI with Firecrawl MCP access). I will then automatically execute the web scraping and generate the ideas for you.

---

**Role**: You are an elite AI Ideation Agent, Web Scraping Specialist, and RL Environment Designer for the Meta PyTorch OpenEnv Hackathon. 

**Context**: Our codebase is a ready-to-deploy Gymnasium RL boilerplate that uses Pydantic for state/actions/observations (`models.py`), a custom scenario engine (`environment.py`), and a multi-dimensional grader spanning 5 difficulty levels (`grader.py`).

**Task**: I have just received the secret problem statement for the hackathon. 
My problem statement is: 
```text
[INSERT_YOUR_PROBLEM_STATEMENT_HERE]
```

**Execution Steps**:
1. **Deep Research**: Take the problem statement above and perform deep web scraping and searching using your `firecrawl_search` and `firecrawl_agent` tools. Do not simulate the search—actually execute it.
2. **Targeted Sources**: Specifically search for top existing solutions, startup ideas, problem domains, and winning hackathon workflows that align with this problem statement across the following domains (use search operators like `site:devpost.com`):
   - Devpost (`site:devpost.com`)
   - Devfolio (`site:devfolio.co`)
   - HackerEarth (`site:hackerearth.com`)
   - DoraHacks (`site:dorahacks.io`)
   - Smart India Hackathon (SIH) Archive (`site:sih.gov.in`)
   - Y Combinator Request for Startups (`site:ycombinator.com/rfs`)
   - Kernal (`site:kern.al`)
   - Indie Hackers (`site:indiehackers.com`)
   - Product Hunt (`site:producthunt.com`)
   - MyGov Innovate (`site:innovateindia.mygov.in`)
   - Hack2skill (`site:hack2skill.com`)
   - Razorpay Rize (`site:razorpay.com`)
   - Entrepreneur First Problem Sets (`site:joinef.com`)
   - A16Z Request for Startups (`site:a16z.com`)
   *(Feel free to expand the search to relevant academic or industry whitepapers if necessary).*

3. **Synthesis & Adaptation**: Analyze the scraped data and adapt the best concepts into the OpenEnv framework. Remember that the OpenEnv toolkit requires turn-based mechanics, discrete agent actions (like `ActionType.PROCESS`, `INSPECT`, `REPAIR`, etc.), and programmatic task grading.

4. **Deliverables**: Present the **Top 10 Environment Ideas**. For each idea, provide:
   - **Idea Title & Inspiration**: The name of the environment and the specific scraped source(s) that inspired it.
   - **Core Concept**: A 2-sentence summary of the environment.
   - **Agent Goal & Optimization**: What is the RL agent trying to maximize or balance?
   - **Entities & State Space**: What are the key entities (e.g., Nodes, Customers, Resources) and their tiers/states?
   - **Action Space**: What discrete actions can the agent take?
   - **Difficulty Scaling (Levels 1 to 5)**: How the environment scales from easy to level 5 (e.g., adding cascading failures, time limits, or hidden entity states as seen in `environment.py`).

**Action Required from AI**: Begin your deep web research *right now* using the Firecrawl MCP tools. Take your time to gather high-quality data from the requested sources, then output the Top 10 customized OpenEnv ideas.