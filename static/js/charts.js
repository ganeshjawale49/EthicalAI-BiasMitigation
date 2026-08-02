/* Chart.js Visualization Utility Functions for Ethical AI System */

function renderSelectionRateChart(canvasId, privLabel, unprivLabel, privRate, unprivRate) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [privLabel + ' (Privileged)', unprivLabel + ' (Unprivileged)'],
            datasets: [{
                label: 'Positive Prediction Rate (Approval %)',
                data: [(privRate * 100).toFixed(1), (unprivRate * 100).toFixed(1)],
                backgroundColor: ['#2563eb', '#ef4444'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Demographic Selection Rate Disparity' }
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Selection Rate (%)' } }
            }
        }
    });
}

function renderMitigationComparisonChart(canvasId, beforeDI, afterDI, beforeAcc, afterAcc) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Disparate Impact (Fairness)', 'Model Accuracy (%)'],
            datasets: [
                {
                    label: 'Before Mitigation (Biased)',
                    data: [beforeDI.toFixed(2), (beforeAcc * 100).toFixed(1)],
                    backgroundColor: '#ef4444',
                    borderRadius: 6
                },
                {
                    label: 'After Mitigation (Fair)',
                    data: [afterDI.toFixed(2), (afterAcc * 100).toFixed(1)],
                    backgroundColor: '#10b981',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Fairness & Accuracy Performance Trade-off' }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
