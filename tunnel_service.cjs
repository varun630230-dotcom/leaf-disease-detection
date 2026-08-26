const localtunnel = require('localtunnel');

(async () => {
  async function connect() {
    try {
      const tunnel = await localtunnel({ port: 5173, subdomain: 'leafguard-ai-preview' });
      console.log('Tunnel URL:', tunnel.url);

      tunnel.on('close', () => {
        console.log('Tunnel closed. Reconnecting in 3s...');
        setTimeout(connect, 3000);
      });

      tunnel.on('error', (err) => {
        console.error('Tunnel error:', err);
        tunnel.close();
      });
    } catch (e) {
      console.error('Connection failed, retrying...', e.message);
      setTimeout(connect, 3000);
    }
  }

  connect();
})();
