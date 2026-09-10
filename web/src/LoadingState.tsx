import { Component, useEffect, useState, type ReactNode } from "react";
import { LoaderCircle, RefreshCw, WifiOff } from "lucide-react";

export function LoadingState({ text = "正在准备页面内容…" }: { text?: string }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), 8000);
    return () => clearTimeout(timer);
  }, []);
  return <div className="loading-state" role="status" aria-live="polite" aria-busy="true">
    <div className="loading-state-label"><LoaderCircle size={20} className="spin" /><strong>{text}</strong></div>
    <div className="loading-skeleton" aria-hidden="true"><i /><div><i /><i /><i /></div><i /><i /></div>
    <p>{slow ? "连接较慢，仍在加载中。你可以切换页面，或稍后重试。" : "正在获取当前页面所需的研究数据"}</p>
  </div>;
}

export class PageBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <div className="error-panel" role="alert"><div><strong>页面暂时未能载入</strong><p>请检查网络后重试，你保存的研究清单不会丢失。</p><button className="button secondary" onClick={() => location.reload()}><RefreshCw size={16} />重新加载页面</button></div></div>;
    return this.props.children;
  }
}

export function ConnectionStatus() {
  const [offline, setOffline] = useState(!navigator.onLine);
  useEffect(() => {
    const update = () => setOffline(!navigator.onLine);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => { window.removeEventListener("online", update); window.removeEventListener("offline", update); };
  }, []);
  return offline ? <div className="connection-status" role="status"><WifiOff size={16} />网络已断开，已载入的内容仍可查看。恢复连接后可重试。</div> : null;
}
