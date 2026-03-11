// Dashboard WebSocket connections and Chart.js setup

const MAX_CHART_POINTS = 120;

let chart = null;
let videoWs = null;
let metricsWs = null;

function initChart() {
    const ctx = document.getElementById("metrics-chart").getContext("2d");
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Queue Count",
                    data: [],
                    borderColor: "#e94560",
                    backgroundColor: "rgba(233, 69, 96, 0.1)",
                    tension: 0.3,
                    fill: true,
                    yAxisID: "y",
                },
                {
                    label: "Wait Time (s)",
                    data: [],
                    borderColor: "#4caf50",
                    backgroundColor: "rgba(76, 175, 80, 0.1)",
                    tension: 0.3,
                    fill: true,
                    yAxisID: "y1",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { labels: { color: "#aaa" } },
            },
            scales: {
                x: {
                    ticks: { color: "#666", maxTicksLimit: 10 },
                    grid: { color: "rgba(255,255,255,0.05)" },
                },
                y: {
                    type: "linear",
                    position: "left",
                    title: { display: true, text: "People", color: "#aaa" },
                    ticks: { color: "#e94560" },
                    grid: { color: "rgba(255,255,255,0.05)" },
                    beginAtZero: true,
                },
                y1: {
                    type: "linear",
                    position: "right",
                    title: { display: true, text: "Wait (s)", color: "#aaa" },
                    ticks: { color: "#4caf50" },
                    grid: { drawOnChartArea: false },
                    beginAtZero: true,
                },
            },
        },
    });
}

function connectVideoWs() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    videoWs = new WebSocket(`${protocol}//${location.host}/ws/video`);
    videoWs.binaryType = "arraybuffer";

    const img = document.getElementById("video-feed");
    const statusDot = document.getElementById("video-status");

    videoWs.onopen = () => {
        statusDot.classList.remove("disconnected");
    };

    videoWs.onmessage = (event) => {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        img.onload = () => URL.revokeObjectURL(url);
        img.src = url;
    };

    videoWs.onclose = () => {
        statusDot.classList.add("disconnected");
        setTimeout(connectVideoWs, 3000);
    };
}

function connectMetricsWs() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    metricsWs = new WebSocket(`${protocol}//${location.host}/ws/metrics`);

    const statusDot = document.getElementById("metrics-status");

    metricsWs.onopen = () => {
        statusDot.classList.remove("disconnected");
    };

    metricsWs.onmessage = (event) => {
        const metrics = JSON.parse(event.data);
        if (!metrics || metrics.length === 0) return;

        const m = metrics[0]; // primary zone
        document.getElementById("count-value").textContent = m.smoothed_count;
        document.getElementById("wait-value").textContent = `${Math.round(m.wait_time)}s`;
        document.getElementById("mode-value").textContent = m.estimation_mode;
        document.getElementById("service-value").textContent = `${m.service_time}s`;

        // Update chart
        const timeLabel = new Date(m.timestamp * 1000).toLocaleTimeString();
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(m.smoothed_count);
        chart.data.datasets[1].data.push(m.wait_time);

        if (chart.data.labels.length > MAX_CHART_POINTS) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }

        chart.update();
    };

    metricsWs.onclose = () => {
        statusDot.classList.add("disconnected");
        setTimeout(connectMetricsWs, 3000);
    };
}

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    connectVideoWs();
    connectMetricsWs();
});
