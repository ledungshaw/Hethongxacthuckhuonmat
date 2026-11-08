const API_BASE = '';

function updateUI(result) {
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const statusMessage = document.getElementById('status-message');

    statusCard.className = 'status-card';

    if (result.status === 'gate_opened' || result.status === 'confirmed') {
        statusCard.classList.add('gate_opened');
        statusIcon.textContent = 'OK';
    } else if (result.status === 'spoof_detected') {
        statusCard.classList.add('spoof_detected');
        statusIcon.textContent = '!';
    } else if (result.status === 'unknown') {
        statusCard.classList.add('unknown');
        statusIcon.textContent = 'X';
    } else {
        statusIcon.textContent = '...';
    }

    statusMessage.textContent = result.message || 'Dang cho...';
    document.getElementById('user-name').textContent = result.user_name || '-';
    document.getElementById('user-id').textContent = result.user_id || '-';
    document.getElementById('similarity').textContent = (result.similarity || 0).toFixed(4);

    if (result.is_live !== undefined) {
        const liveness = result.is_live
            ? `THAT (${Math.round((result.liveness_score || 0) * 100)}%)`
            : `GIA (${Math.round((result.liveness_score || 0) * 100)}%)`;
        document.getElementById('liveness').textContent = liveness;
    }

    document.getElementById('timestamp').textContent = result.timestamp || '-';
    document.getElementById('fps').textContent = result.fps || 0;
}

async function pollResult() {
    try {
        const res = await fetch(`${API_BASE}/api/current_result`);
        const result = await res.json();
        updateUI(result);
    } catch (e) { console.error('Poll error:', e); }
}

async function pollStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const status = await res.json();
        document.getElementById('faces').textContent = status.faces_loaded || 0;
    } catch (e) { console.error('Status error:', e); }
}

async function pollHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        const history = await res.json();
        const tbody = document.getElementById('history-body');

        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#999;">Chua co du lieu</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(item => `
            <tr>
                <td>${item.time}</td>
                <td>${item.user_id}</td>
                <td>${item.user_name}</td>
                <td>${item.similarity.toFixed(4)}</td>
                <td>${item.is_live ? 'OK ' + item.action : 'GIA'}</td>
            </tr>
        `).join('');
    } catch (e) { console.error('History error:', e); }
}

async function syncFaces(event) {
    try {
        const btn = event.target;
        btn.textContent = 'Dang sync...';
        btn.disabled = true;

        const res = await fetch(`${API_BASE}/api/sync`, { method: 'POST' });
        const data = await res.json();

        btn.textContent = data.success ? `Da sync ${data.count}` : 'Loi';

        setTimeout(() => {
            btn.textContent = 'Sync';
            btn.disabled = false;
        }, 2000);
    } catch (e) { console.error('Sync error:', e); }
}

setInterval(pollResult, 200);
setInterval(pollStatus, 2000);
setInterval(pollHistory, 3000);

pollResult();
pollStatus();
pollHistory();
