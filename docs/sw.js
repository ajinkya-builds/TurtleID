const SHELL_CACHE = 'turtleid-shell-v2';
const MODEL_CACHE = 'turtleid-onnx-v1';
const SHELL_ASSETS = [
  './',
  './index.html',
  './stack_v2_client.js',
  './manifest.json',
  './data/stack_v2_browser_bundle.json'
];

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(SHELL_CACHE).then(function (cache) {
      return cache.addAll(SHELL_ASSETS).catch(function () {});
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(
        keys.map(function (key) {
          if (key !== SHELL_CACHE && key !== MODEL_CACHE) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function (event) {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  if (url.pathname.endsWith('.onnx')) {
    event.respondWith(
      caches.open(MODEL_CACHE).then(function (cache) {
        return cache.match(req).then(function (cached) {
          if (cached) return cached;
          return fetch(req).then(function (res) {
            if (res && res.ok) cache.put(req, res.clone());
            return res;
          });
        });
      })
    );
    return;
  }

  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(req).then(function (cached) {
        const net = fetch(req).then(function (res) {
          if (res && res.ok) {
            caches.open(SHELL_CACHE).then(function (cache) {
              cache.put(req, res.clone());
            });
          }
          return res;
        }).catch(function () {
          return cached;
        });
        return cached || net;
      })
    );
  }
});
