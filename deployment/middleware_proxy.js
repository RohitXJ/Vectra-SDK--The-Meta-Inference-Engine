export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // 1. Get the target backend URL from environment variables
    // Example: https://username-space.hf.space
    const backendBaseUrl = env.API_BASE_URL;
    
    if (!backendBaseUrl) {
      return new Response("Cloudflare Worker Error: API_BASE_URL not set", { status: 500 });
    }

    // 2. Construct the new target URL
    const targetUrl = backendBaseUrl + url.pathname + url.search;

    // 3. Prepare the new request with the hidden Token
    const newRequest = new Request(targetUrl, {
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body,
      redirect: 'follow'
    });

    // 4. Inject the Authorization Header securely (never seen by browser)
    if (env.HF_TOKEN) {
      newRequest.headers.set('Authorization', `Bearer ${env.HF_TOKEN}`);
    }

    // 5. Handle CORS Preflight (OPTIONS)
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': '*',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // 6. Forward the request and return the response
    try {
      const response = await fetch(newRequest);
      
      // Clone the response so we can modify headers if needed (for CORS)
      const newResponse = new Response(response.body, response);
      newResponse.headers.set('Access-Control-Allow-Origin', '*');
      
      return newResponse;
    } catch (e) {
      return new Response("Proxy Error: " + e.message, { status: 502 });
    }
  }
};
