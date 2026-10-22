const http = require('http');
const fs = require('fs');
const path = require('path');

const LOG_DIR = path.join(__dirname, 'chat_logs');
const PORT = 3001;

// Создаём папку для логов если нет
if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
}

const server = http.createServer((req, res) => {
    // CORS заголовки
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // Сохранить лог чата
    if (req.method === 'POST' && req.url === '/save-log') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const chatId = data.chatId || 'unknown';
                const chatName = (data.chatName || 'chat')
                    .replace(/[^a-zA-Zа-яА-Я0-9_\- ]/g, '')
                    .substring(0, 50);
                const filename = chatId + '_' + chatName + '.json';
                const filepath = path.join(LOG_DIR, filename);

                fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf8');

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    ok: true,
                    path: filepath
                }));
            } catch (e) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    ok: false,
                    error: e.message
                }));
            }
        });
        return;
    }

    // Сохранить полную историю всех чатов
    if (req.method === 'POST' && req.url === '/save-all') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const filepath = path.join(LOG_DIR, 'all_chats_backup.json');
                fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf8');

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: true, path: filepath }));
            } catch (e) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: false, error: e.message }));
            }
        });
        return;
    }

    // Получить список логов
    if (req.method === 'GET' && req.url === '/list-logs') {
        try {
            const files = fs.readdirSync(LOG_DIR)
                .filter(f => f.endsWith('.json') && f !== 'all_chats_backup.json')
                .map(f => {
                    const stat = fs.statSync(path.join(LOG_DIR, f));
                    return {
                        name: f,
                        size: stat.size,
                        modified: stat.mtime
                    };
                });
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ ok: true, files: files }));
        } catch (e) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ ok: false, error: e.message }));
        }
        return;
    }

    // Загрузить конкретный лог
    if (req.method === 'GET' && req.url.startsWith('/load-log/')) {
        try {
            const filename = decodeURIComponent(
                req.url.replace('/load-log/', '')
            );
            // Защита от path traversal
            const safe = path.basename(filename);
            const filepath = path.join(LOG_DIR, safe);

            if (fs.existsSync(filepath)) {
                const content = fs.readFileSync(filepath, 'utf8');
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(content);
            } else {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: false, error: 'Not found' }));
            }
        } catch (e) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ ok: false, error: e.message }));
        }
        return;
    }

    // Удалить лог
    if (req.method === 'POST' && req.url === '/delete-log') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const safe = path.basename(data.filename || '');
                const filepath = path.join(LOG_DIR, safe);

                if (fs.existsSync(filepath)) {
                    fs.unlinkSync(filepath);
                }
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: true }));
            } catch (e) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: false, error: e.message }));
            }
        });
        return;
    }

    res.writeHead(404);
    res.end('Not found');
});

server.listen(PORT, () => {
    console.log('Log server running on http://localhost:' + PORT);
    console.log('Logs directory: ' + LOG_DIR);
});