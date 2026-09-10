import { useEffect, useRef, useState, type ReactNode } from "react";
import "./motion.css";

export function useReducedMotion() {
  const [reduced, setReduced] = useState(() =>
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

export function AnimatedNumber({ value, decimals = 0, duration = 650, className = "" }: {
  value: number; decimals?: number; duration?: number; className?: string;
}) {
  const reduced = useReducedMotion();
  const [display, setDisplay] = useState(reduced ? value : 0);
  const previous = useRef(0);
  useEffect(() => {
    if (reduced || !Number.isFinite(value)) { setDisplay(value); previous.current = value; return; }
    const from = previous.current;
    let start = 0;
    let frame = 0;
    const tick = (now: number) => {
      if (!start) start = now;
      const progress = Math.min(1, (now - start) / Math.max(1, duration));
      const next = from + (value - from) * (1 - (1 - progress) ** 3);
      setDisplay(next);
      previous.current = next;
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration, reduced]);
  const format = (number: number) => number.toLocaleString("zh-CN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return <span className={`bm-animated-number ${className}`} aria-label={format(value)}><span aria-hidden="true">{format(display)}</span></span>;
}

export function Reveal({ children, className = "", delay = 0 }: {
  children: ReactNode; className?: string; delay?: number;
}) {
  return <div className={`bm-reveal ${className}`} style={{ animationDelay: `${delay}ms` }}>{children}</div>;
}
