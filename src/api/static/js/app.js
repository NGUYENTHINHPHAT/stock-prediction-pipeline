// ===== State =====
let trainingPollInterval = null;

// ===== Init =====
let backtest = null;

document.addEventListener('DOMContentLoaded', () => {
    refreshStatus();
    initStockChips();
    loadBacktest();
});

// ===== Stock Chip Selection =====
function initStockChips() {
    document.querySelectorAll('.stock-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chip.classList.toggle('selected');
            document.getElementById('chips-error').classList.add('hidden');
        });
    });
}

function getSelectedSymbols() {
    return Array.from(document.querySelectorAll('.stock-chip.selected'))
        .map(c => c.dataset.symbol);
}

// ===== Error Handling =====
function showError(msg) {
    const banner = document.getElementById('error-banner');
    document.getElementById('error-message').textContent = msg;
    banner.classList.remove('hidden');
    setTimeout(dismissError, 8000);
}

function dismissError() {
    document.getElementById('error-banner').classList.add('hidden');
}

// ===== Model Status =====
async function refreshStatus() {
    try {
        const res = await fetch('/');
        const data = await res.json();

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const trained = document.getElementById('last-trained');

        if (data.model_loaded) {
            dot.className = 'status-dot green';
            text.textContent = 'Model loaded';
        } else {
            dot.className = 'status-dot red';
            text.textContent = 'Not loaded';
        }

        trained.textContent = data.last_trained
            ? new Date(data.last_trained).toLocaleString()
            : 'Never';
    } catch (e) {
        // Silently fail on status check
    }
}

// ===== Training =====
async function startTraining() {
    const btn = document.getElementById('train-btn');
    const btnText = document.getElementById('train-btn-text');
    const spinner = document.getElementById('train-spinner');
    const statusMsg = document.getElementById('training-status-msg');

    btn.disabled = true;
    btnText.textContent = 'Training...';
    spinner.classList.remove('hidden');
    statusMsg.classList.remove('hidden');
    statusMsg.textContent = 'Training started. This may take a few minutes...';

    try {
        const res = await fetch('/train', { method: 'POST' });
        const data = await res.json();

        if (data.status === 'skipped') {
            statusMsg.textContent = data.message;
            btn.disabled = false;
            btnText.textContent = 'Train Model';
            spinner.classList.add('hidden');
            return;
        }

        // Poll for completion
        trainingPollInterval = setInterval(pollTrainingStatus, 3000);
    } catch (e) {
        showError('Failed to start training: ' + e.message);
        btn.disabled = false;
        btnText.textContent = 'Train Model';
        spinner.classList.add('hidden');
        statusMsg.classList.add('hidden');
    }
}

async function pollTrainingStatus() {
    try {
        const res = await fetch('/training/status');
        const data = await res.json();

        const statusMsg = document.getElementById('training-status-msg');
        statusMsg.textContent = data.message;

        if (data.status === 'complete' || data.status === 'failed') {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;

            const btn = document.getElementById('train-btn');
            const btnText = document.getElementById('train-btn-text');
            const spinner = document.getElementById('train-spinner');

            btn.disabled = false;
            btnText.textContent = 'Train Model';
            spinner.classList.add('hidden');

            if (data.status === 'complete') {
                refreshStatus();
            } else {
                showError('Training failed: ' + data.message);
            }
        }
    } catch (e) {
        // Keep polling on transient errors
    }
}

// ===== Predictions =====
async function submitPrediction(event) {
    event.preventDefault();

    const symbols = getSelectedSymbols();
    if (symbols.length === 0) {
        document.getElementById('chips-error').classList.remove('hidden');
        return;
    }

    const btn = document.getElementById('predict-btn');
    const btnText = document.getElementById('predict-btn-text');
    const spinner = document.getElementById('predict-spinner');

    btn.disabled = true;
    btnText.textContent = 'Loading...';
    spinner.classList.remove('hidden');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await res.json();
        renderResults(data);
    } catch (e) {
        showError(e.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Get Predictions';
        spinner.classList.add('hidden');
    }
}

function renderResults(data) {
    const section = document.getElementById('results-section');
    const meta = document.getElementById('results-meta');
    const note = document.getElementById('results-note');
    const grid = document.getElementById('results-grid');

    section.classList.remove('hidden');

    const modelInfo = data.model_name || 'Unknown';
    const trainedAt = data.model_last_trained
        ? new Date(data.model_last_trained).toLocaleString()
        : 'Unknown';
    const horizon = backtest ? backtest.horizon_days : 5;
    meta.textContent = `Model: ${modelInfo} | Last trained: ${trainedAt} | Horizon: ${horizon} trading days`;

    note.textContent = backtest
        ? `Read P(up) against each stock's historical base rate: in the backtest these stocks rose in `
          + `${pct(backtest.base_rate)} of ${horizon}-day windows, so 55–60% is the default, not a signal. `
          + (isSignificant(backtest)
              ? `The backtest found a statistically significant edge, but check its size before relying on it.`
              : `The backtest found no significant edge, so treat rank and tilt as illustrative.`)
        : '';

    const n = data.predictions.length;
    const sorted = [...data.predictions].sort((a, b) => a.rank - b.rank);

    grid.innerHTML = sorted.map(p => {
        const edgePp = p.edge * 100;
        const edgeSign = edgePp >= 0 ? '+' : '−';
        const edgeClass = Math.abs(edgePp) < 3 ? 'flat' : (edgePp > 0 ? 'up' : 'down');
        const signalClass = p.signal === 'no edge' ? 'signal-none'
            : (p.signal.startsWith('weak') ? 'signal-weak' : 'signal-moderate');
        const barPos = Math.max(0, Math.min(100, p.p_up * 100));
        const basePos = Math.max(0, Math.min(100, p.base_rate * 100));

        return `
            <div class="result-card">
                <div class="result-card-header">
                    <span class="result-symbol">${p.symbol}</span>
                    <span class="rank-badge" title="Ranked by P(up) among selected stocks">#${p.rank} of ${n}</span>
                </div>
                <div class="result-price">Current: $${p.current_price.toFixed(2)}</div>
                <div class="pup-row">
                    <span class="probability">${(p.p_up * 100).toFixed(1)}%</span>
                    <span class="pup-label">P(up)</span>
                </div>
                <div class="edge-meter" title="Marker = P(up); tick = historical base rate">
                    <div class="edge-track"></div>
                    <div class="edge-base" style="left:${basePos}%"></div>
                    <div class="edge-dot ${edgeClass}" style="left:${barPos}%"></div>
                </div>
                <div class="result-details">
                    <span class="edge-text">
                        <strong class="edge-${edgeClass}">${edgeSign}${Math.abs(edgePp).toFixed(1)} pp</strong>
                        vs ${pct(p.base_rate)} base rate
                    </span>
                    <span class="confidence-badge ${signalClass}">${p.signal}</span>
                </div>
            </div>
        `;
    }).join('');
}

// ===== Backtest panel =====
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const SERIES_STYLE = {
    'Buy & hold (equal weight)': { cls: 's1', short: 'Buy & hold', tag: 'Buy & hold' },
    'XGBoost long/flat': { cls: 's2', short: 'XGBoost long/flat', tag: 'Long/flat' },
    'XGBoost long/short rank (mkt-neutral)': { cls: 's3', short: 'XGBoost long/short (mkt-neutral)', tag: 'Long/short' },
};

async function loadBacktest() {
    try {
        const res = await fetch('/backtest');
        if (!res.ok) return;
        backtest = await res.json();
        renderBacktest(backtest);
    } catch (e) {
        // Panel stays hidden if the backtest isn't available
    }
}

// An edge counts only if a test is significant AND points the right way.
function isSignificant(bt) {
    const auc = bt.permutation_test, ic = bt.ic_test;
    return (auc.p_value < 0.05 && auc.auc > 0.5) || (ic && ic.p_value < 0.05 && ic.mean_ic > 0);
}

function findRow(rows, key, name) {
    return rows.find(r => r[key] === name) || {};
}

function renderBacktest(bt) {
    document.getElementById('backtest-section').classList.remove('hidden');

    const [start, end] = bt.oos_period;
    document.getElementById('bt-meta').textContent =
        `${start} → ${end} · ${bt.symbols.length} stocks · ${bt.horizon_days}-day horizon · ${bt.cost_bps} bps costs`;

    const xgbClf = findRow(bt.classification, 'model', 'XGBoost (tuned)');
    const upClf = findRow(bt.classification, 'model', 'Always UP (baseline)');
    const xgbStrat = findRow(bt.strategies, 'strategy', 'XGBoost long/flat');
    const bhStrat = findRow(bt.strategies, 'strategy', 'Buy & hold (equal weight)');
    const aucP = bt.permutation_test.p_value;
    const ic = bt.ic_test || { mean_ic: 0, p_value: 1 };
    const boot = bt.sharpe_diff_bootstrap;
    const significant = isSignificant(bt);

    const verdict = document.getElementById('bt-verdict');
    verdict.className = 'verdict ' + (significant ? 'verdict-pass' : 'verdict-fail');
    verdict.innerHTML = significant
        ? `<strong>Verdict: statistically significant edge</strong>
           (overall AUC p = ${aucP.toFixed(3)}, stock-picking IC p = ${ic.p_value.toFixed(3)}). Check its size and costs below.`
        : `<strong>Verdict: no statistically significant edge.</strong>
           Neither overall skill (AUC p = ${aucP.toFixed(2)}) nor stock-picking skill (rank IC p = ${ic.p_value.toFixed(2)})
           can be told apart from random, and the model doesn't beat simply predicting "UP" every week. That is the expected
           result for public technical signals on the most heavily traded stocks in the world, and this page shows the evidence instead of hiding it.`;

    const kpis = [
        { label: 'Accuracy', value: pct(xgbClf.accuracy), sub: `vs ${pct(upClf.accuracy)} always-UP baseline`,
          bad: xgbClf.accuracy <= upClf.accuracy },
        { label: 'AUC (overall skill)', value: xgbClf.auc.toFixed(3),
          sub: `p = ${aucP.toFixed(2)} · stock-picking IC ${ic.mean_ic >= 0 ? '+' : '−'}${Math.abs(ic.mean_ic).toFixed(3)} (p = ${ic.p_value.toFixed(2)})`,
          bad: !significant },
        { label: 'Net Sharpe, long/flat', value: xgbStrat.sharpe.toFixed(2),
          sub: `vs ${bhStrat.sharpe.toFixed(2)} buy & hold · Δ 95% CI [${boot.ci_low.toFixed(2)}, ${boot.ci_high.toFixed(2)}]`,
          bad: boot.ci_low <= 0 },
        { label: 'CAGR, long/flat', value: pct(xgbStrat.cagr), sub: `vs ${pct(bhStrat.cagr)} buy & hold · ${pct(xgbStrat.avg_exposure, 0)} invested`,
          bad: xgbStrat.cagr < bhStrat.cagr },
    ];
    document.getElementById('bt-kpis').innerHTML = kpis.map(k => `
        <div class="kpi">
            <div class="kpi-label">${k.label}</div>
            <div class="kpi-value">${k.value}</div>
            <div class="kpi-sub"><span class="kpi-flag ${k.bad ? 'flag-bad' : 'flag-good'}">${k.bad ? '✕' : '✓'}</span>${k.sub}</div>
        </div>`).join('');

    // Classification table
    const clfRows = bt.classification.map(r => `
        <tr class="${r.model.startsWith('XGBoost') ? 'row-hl' : ''}">
            <td>${r.model}</td><td class="num">${pct(r.accuracy)}</td><td class="num">${r.auc.toFixed(3)}</td>
        </tr>`).join('');
    document.getElementById('bt-clf-table').innerHTML =
        `<thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">AUC</th></tr></thead><tbody>${clfRows}</tbody>`;

    const stratRows = bt.strategies.map(r => `
        <tr class="${r.strategy === 'XGBoost long/flat' ? 'row-hl' : ''}">
            <td>${r.strategy}</td><td class="num">${r.sharpe.toFixed(2)}</td>
            <td class="num">${pct(r.cagr)}</td><td class="num">${pct(r.max_drawdown)}</td>
        </tr>`).join('');
    document.getElementById('bt-strat-table').innerHTML =
        `<thead><tr><th>Strategy</th><th class="num">Sharpe</th><th class="num">CAGR</th><th class="num">Max DD</th></tr></thead><tbody>${stratRows}</tbody>`;

    document.getElementById('bt-method').textContent =
        `Method: ${bt.n_folds}-fold expanding walk-forward split by date, with a ${bt.horizon_days}-day purge so no training label `
        + `overlaps the test period. Positions rebalance every ${bt.horizon_days} trading days (non-overlapping), long when P(up) > `
        + `${bt.threshold}, costs ${bt.cost_bps} bps per unit turnover. Significance: permutation test on AUC (overall skill), `
        + `permutation test on mean weekly rank IC with returns shuffled within each date (stock-picking skill), `
        + `and a block bootstrap on the Sharpe difference vs buy & hold.`;

    renderEquityChart(bt.equity);
}

function renderEquityChart(eq) {
    const wrap = document.getElementById('bt-chart');
    const tooltip = document.getElementById('bt-tooltip');
    const names = Object.keys(eq.series);
    const dates = eq.dates.map(d => new Date(d + 'T00:00:00'));
    const n = dates.length;

    // Legend (identity is never color-alone: legend + direct end labels)
    document.getElementById('bt-legend').innerHTML = names.map(nm =>
        `<span class="legend-item"><span class="legend-swatch ${SERIES_STYLE[nm]?.cls || ''}"></span>${SERIES_STYLE[nm]?.short || nm}</span>`
    ).join('');

    const W = 880, H = 300, m = { t: 12, r: 120, b: 28, l: 48 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const all = names.flatMap(nm => eq.series[nm]);
    const yMin = Math.min(0.5, Math.floor(Math.min(...all) * 4) / 4);
    const yMax = Math.ceil(Math.max(...all) * 2) / 2;
    const x = i => m.l + (i / (n - 1)) * iw;
    const y = v => m.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('class', 'equity-svg');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Out-of-sample growth of one dollar for buy and hold versus XGBoost strategies');
    const el = (tag, attrs, parent = svg) => {
        const e = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        parent.appendChild(e);
        return e;
    };

    // Grid + y labels
    const step = (yMax - yMin) > 2 ? 0.5 : 0.25;
    for (let v = yMin; v <= yMax + 1e-9; v += step) {
        el('line', { x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v), class: v === 1 ? 'grid-line grid-base' : 'grid-line' });
        el('text', { x: m.l - 8, y: y(v) + 4, class: 'axis-label', 'text-anchor': 'end' }).textContent = `$${v.toFixed(2)}`;
    }
    // X ticks: each January
    let lastYear = null;
    dates.forEach((d, i) => {
        if (d.getFullYear() !== lastYear) {
            if (lastYear !== null) {
                el('line', { x1: x(i), x2: x(i), y1: m.t + ih, y2: m.t + ih + 4, class: 'axis-tick' });
                el('text', { x: x(i), y: H - 8, class: 'axis-label', 'text-anchor': 'middle' }).textContent = d.getFullYear();
            }
            lastYear = d.getFullYear();
        }
    });

    // Lines + direct end labels (nudged apart to avoid collisions)
    const ends = names.map(nm => ({ nm, v: eq.series[nm][n - 1], y: y(eq.series[nm][n - 1]) }))
        .sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
    names.forEach(nm => {
        const d = eq.series[nm].map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
        el('path', { d, class: `series-line ${SERIES_STYLE[nm]?.cls || ''}` });
    });
    ends.forEach(e => {
        const t = el('text', { x: m.l + iw + 8, y: e.y + 4, class: 'end-label' });
        t.textContent = `${SERIES_STYLE[e.nm]?.tag || e.nm} $${e.v.toFixed(2)}`;
    });

    // Crosshair layer
    const hair = el('line', { y1: m.t, y2: m.t + ih, class: 'crosshair hidden' });
    const dots = names.map(nm => el('circle', { r: 4.5, class: `hover-dot ${SERIES_STYLE[nm]?.cls || ''} hidden` }));
    const hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent', tabindex: 0 });

    const show = i => {
        hair.setAttribute('x1', x(i)); hair.setAttribute('x2', x(i));
        hair.classList.remove('hidden');
        names.forEach((nm, k) => {
            dots[k].setAttribute('cx', x(i)); dots[k].setAttribute('cy', y(eq.series[nm][i]));
            dots[k].classList.remove('hidden');
        });
        tooltip.textContent = '';
        const head = document.createElement('div');
        head.className = 'tt-head';
        head.textContent = dates[i].toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
        tooltip.appendChild(head);
        names.map(nm => ({ nm, v: eq.series[nm][i] })).sort((a, b) => b.v - a.v).forEach(({ nm, v }) => {
            const row = document.createElement('div');
            row.className = 'tt-row';
            const sw = document.createElement('span');
            sw.className = `legend-swatch ${SERIES_STYLE[nm]?.cls || ''}`;
            const val = document.createElement('strong');
            val.textContent = `$${v.toFixed(2)}`;
            const lab = document.createElement('span');
            lab.textContent = SERIES_STYLE[nm]?.short || nm;
            row.append(sw, val, lab);
            tooltip.appendChild(row);
        });
        tooltip.classList.remove('hidden');
        const box = wrap.getBoundingClientRect();
        const px = (x(i) / W) * box.width;
        const left = px > box.width * 0.6 ? px - tooltip.offsetWidth - 14 : px + 14;
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `8px`;
    };
    const hide = () => {
        hair.classList.add('hidden');
        dots.forEach(d => d.classList.add('hidden'));
        tooltip.classList.add('hidden');
    };
    let cur = n - 1;
    hit.addEventListener('pointermove', ev => {
        const r = svg.getBoundingClientRect();
        const sx = ((ev.clientX - r.left) / r.width) * W;
        cur = Math.max(0, Math.min(n - 1, Math.round(((sx - m.l) / iw) * (n - 1))));
        show(cur);
    });
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', () => show(cur));
    hit.addEventListener('blur', hide);
    hit.addEventListener('keydown', ev => {
        if (ev.key === 'ArrowLeft') { cur = Math.max(0, cur - 1); show(cur); ev.preventDefault(); }
        if (ev.key === 'ArrowRight') { cur = Math.min(n - 1, cur + 1); show(cur); ev.preventDefault(); }
    });

    wrap.querySelector('svg')?.remove();
    wrap.insertBefore(svg, tooltip);
}
