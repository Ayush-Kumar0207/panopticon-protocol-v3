import { useState, useEffect, useRef } from 'react';

export function useTypewriter(lines, charDelay = 35, lineDelay = 400) {
  const [displayed, setDisplayed] = useState([]);
  const [currentLine, setCurrentLine] = useState(0);
  const [currentChar, setCurrentChar] = useState(0);
  const [done, setDone] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    setDisplayed([]);
    setCurrentLine(0);
    setCurrentChar(0);
    setDone(false);
  }, [lines]);

  useEffect(() => {
    if (done || !lines || lines.length === 0) return;

    if (currentLine >= lines.length) {
      setDone(true);
      return;
    }

    const line = lines[currentLine];

    if (currentChar === 0) {
      setDisplayed(prev => [...prev, '']);
    }

    if (currentChar < line.length) {
      intervalRef.current = setTimeout(() => {
        setDisplayed(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = line.substring(0, currentChar + 1);
          return copy;
        });
        setCurrentChar(c => c + 1);
      }, charDelay);
    } else {
      intervalRef.current = setTimeout(() => {
        setCurrentLine(l => l + 1);
        setCurrentChar(0);
      }, lineDelay);
    }

    return () => clearTimeout(intervalRef.current);
  }, [lines, currentLine, currentChar, done, charDelay, lineDelay]);

  return { displayed, done };
}
