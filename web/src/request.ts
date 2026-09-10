// Short-lived, bounded page cache. A cancelled view never receives a stale response.
const cache = new Map<string, { value: unknown; expires: number; size: number }>();
const pending = new Map<string, Promise<unknown>>();
const TTL = 60_000;
const isLiveSPR = (url: string) => url.startsWith("/api/spr-design") || url.startsWith("/api/spr-expanded") || (url.startsWith("/api/evidence?") && url.includes("section=experiments"));
const MAX_BYTES = 12 * 1024 * 1024;
let cacheBytes = 0;

async function request(url: string): Promise<unknown> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30_000);
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      let response: Response;
      try {
        response = await fetch(url, { signal: controller.signal });
      } catch (error) {
        if (controller.signal.aborted) throw new Error("连接等待超时，请重新加载。已有页面仍可继续浏览。");
        if (attempt === 0 && navigator.onLine) {
          await new Promise(resolve => setTimeout(resolve, 600));
          continue;
        }
        throw new Error(navigator.onLine ? "暂时无法连接，请稍后重试。" : "网络已断开，请检查连接后重试。");
      }
      if (response.status === 401) {
        window.location.replace("/login");
        throw new Error("登录已过期，请重新登录。");
      }
      if (!response.ok) {
        if (attempt === 0 && [502, 503, 504].includes(response.status)) {
          await new Promise(resolve => setTimeout(resolve, 600));
          continue;
        }
        let message = "";
        try { message = (await response.json()).error || ""; } catch { /* Gateway may return HTML. */ }
        throw new Error(message || `请求暂时失败 (${response.status})，请重新加载。`);
      }
      const raw = await response.text();
      let value: unknown;
      try { value = JSON.parse(raw); } catch { throw new Error("收到的数据不完整，请重新加载。"); }
      const size = raw.length * 2;
      if (!isLiveSPR(url) && size <= MAX_BYTES / 2) {
        while (cache.size && (cache.size >= 80 || cacheBytes + size > MAX_BYTES)) {
          const key = cache.keys().next().value!;
          cacheBytes -= cache.get(key)!.size;
          cache.delete(key);
        }
        cache.set(url, { value, expires: Date.now() + TTL, size });
        cacheBytes += size;
      }
      return value;
    }
    throw new Error("连接暂时不可用，请重试。");
  } catch (error) {
    if (controller.signal.aborted) throw new Error("连接等待超时，请重新加载。");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export function api<T>(url: string, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
  const entry = isLiveSPR(url) ? undefined : cache.get(url);
  if (entry && entry.expires <= Date.now()) {
    cacheBytes -= entry.size;
    cache.delete(url);
  }
  let work: Promise<unknown>;
  if (entry && entry.expires > Date.now()) work = Promise.resolve(entry.value);
  else {
    work = pending.get(url) || request(url).finally(() => pending.delete(url));
    pending.set(url, work);
  }
  // Consumers cancel independently; shared in-flight work has its own timeout.
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", abort, { once: true });
    work.then(value => {
      if (signal?.aborted) abort(); else resolve(value as T);
    }, reject).finally(() => signal?.removeEventListener("abort", abort));
  });
}
