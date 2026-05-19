export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const backendBaseUrl = env.API_BASE_URL; 
    const allowedOrigin = env.ALLOWED_ORIGIN; 
    const requestOrigin = request.headers.get("Origin");

    const addCORS = (res) => {
      const newRes = new Response(res.body, res);
      newRes.headers.set('Access-Control-Allow-Origin', requestOrigin || '*');
      newRes.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      newRes.headers.set('Access-Control-Allow-Headers', '*');
      return newRes;
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': requestOrigin || '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': '*',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    if (allowedOrigin && requestOrigin) {
        if (requestOrigin.replace(/\/$/, "") !== allowedOrigin.replace(/\/$/, "")) {
            return addCORS(new Response("Unauthorized Origin", { status: 403 }));
        }
    }

    if (!backendBaseUrl) {
      return addCORS(new Response("Worker Error: API_BASE_URL missing", { status: 500 }));
    }

    // AGGRESSIVE URL CLEANUP:
    // This removes any existing slashes and forces a single-slash connection
    const cleanBase = backendBaseUrl.replace(/\/$/, "");
    const cleanPath = url.pathname.replace(/^\/+/, ""); 
    const targetUrl = `${cleanBase}/${cleanPath}${url.search}`;

    const newRequest = new Request(targetUrl, {
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body,
      redirect: 'follow'
    });

    if (env.HF_TOKEN) {
      newRequest.headers.set('Authorization', `Bearer ${env.HF_TOKEN}`);
    }

    try {
      const response = await fetch(newRequest);
      return addCORS(response);
    } catch (e) {
      return addCORS(new Response("Proxy Error: " + e.message, { status: 502 }));
    }
  }
};
