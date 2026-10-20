/**
 * EcomIQ - E-Commerce Intelligence & Decision Support Platform
 * Ultra-Premium Interactive Application Script
 */

// Embedded Real Data from UCI Online Retail II Analysis
const DATA = {
    total_revenue: 20476260.43,
    total_orders: 40077,
    total_customers: 5878,
    total_products: 4917,
    total_units: 11205148,
    total_countries: 43,
    total_cancellations: 22497,
    cancel_pct: 2.18,
    top_products: [
        { StockCode: "M", Description: "Manual", TotalRevenue: 339226.24, TotalQuantitySold: 9629, UniqueCustomers: 438, TotalInvoices: 784 },
        { StockCode: "22423", Description: "REGENCY CAKESTAND 3 TIER", TotalRevenue: 330590.32, TotalQuantitySold: 26478, UniqueCustomers: 1314, TotalInvoices: 3918 },
        { StockCode: "DOT", Description: "DOTCOM POSTAGE", TotalRevenue: 309854.11, TotalQuantitySold: 1415, UniqueCustomers: 1, TotalInvoices: 1415 },
        { StockCode: "85123A", Description: "WHITE HANGING HEART T-LIGHT HOLDER", TotalRevenue: 257724.71, TotalQuantitySold: 94203, UniqueCustomers: 1490, TotalInvoices: 5365 },
        { StockCode: "85099B", Description: "JUMBO BAG RED WHITE SPOTTY", TotalRevenue: 180569.34, TotalQuantitySold: 96757, UniqueCustomers: 978, TotalInvoices: 3989 },
        { StockCode: "23843", Description: "PAPER CRAFT, LITTLE BIRDIE", TotalRevenue: 168469.60, TotalQuantitySold: 80995, UniqueCustomers: 1, TotalInvoices: 1 },
        { StockCode: "47566", Description: "PARTY BUNTING", TotalRevenue: 148318.28, TotalQuantitySold: 28200, UniqueCustomers: 894, TotalInvoices: 2674 },
        { StockCode: "84879", Description: "ASSORTED COLOUR BIRD ORNAMENT", TotalRevenue: 129324.49, TotalQuantitySold: 80082, UniqueCustomers: 1010, TotalInvoices: 2807 },
        { StockCode: "POST", Description: "POSTAGE", TotalRevenue: 125682.42, TotalQuantitySold: 5363, UniqueCustomers: 405, TotalInvoices: 1851 },
        { StockCode: "22086", Description: "PAPER CHAIN KIT 50'S CHRISTMAS", TotalRevenue: 117760.29, TotalQuantitySold: 35084, UniqueCustomers: 896, TotalInvoices: 2018 }
    ],
    top_customers: [
        { CustomerID: 18102, TotalSpend: 580987.04, TotalOrders: 145, TotalUnits: 181645, UniqueProducts: 382, Segment: "VIP Enterprise" },
        { CustomerID: 14646, TotalSpend: 528602.52, TotalOrders: 151, TotalUnits: 367193, UniqueProducts: 961, Segment: "VIP Enterprise" },
        { CustomerID: 14156, TotalSpend: 313437.62, TotalOrders: 156, TotalUnits: 164325, UniqueProducts: 1446, Segment: "VIP Wholesale" },
        { CustomerID: 14911, TotalSpend: 291420.81, TotalOrders: 398, TotalUnits: 147972, UniqueProducts: 2550, Segment: "High Velocity" },
        { CustomerID: 17450, TotalSpend: 244784.25, TotalOrders: 51, TotalUnits: 83914, UniqueProducts: 144, Segment: "High Value" },
        { CustomerID: 13694, TotalSpend: 195640.69, TotalOrders: 143, TotalUnits: 188201, UniqueProducts: 896, Segment: "High Value" },
        { CustomerID: 17511, TotalSpend: 172132.87, TotalOrders: 60, TotalUnits: 117174, UniqueProducts: 657, Segment: "Loyal Growth" },
        { CustomerID: 16446, TotalSpend: 168472.50, TotalOrders: 2, TotalUnits: 80997, UniqueProducts: 3, Segment: "Bulk Buyer" },
        { CustomerID: 16684, TotalSpend: 147142.77, TotalOrders: 55, TotalUnits: 104810, UniqueProducts: 184, Segment: "Regular Wholesale" },
        { CustomerID: 12415, TotalSpend: 144458.37, TotalOrders: 28, TotalUnits: 91447, UniqueProducts: 498, Segment: "High Value" }
    ],
    top_countries: [
        { Country: "United Kingdom", revenue: 17410196.10, orders: 36535, customers: 5350 },
        { Country: "EIRE", revenue: 658767.31, orders: 626, customers: 5 },
        { Country: "Netherlands", revenue: 554038.09, orders: 228, customers: 22 },
        { Country: "Germany", revenue: 431262.51, orders: 893, customers: 107 },
        { Country: "France", revenue: 347043.19, orders: 615, customers: 95 },
        { Country: "Australia", revenue: 169968.17, orders: 81, customers: 15 },
        { Country: "Switzerland", revenue: 100913.67, orders: 112, customers: 22 },
        { Country: "Spain", revenue: 91873.97, orders: 135, customers: 30 },
        { Country: "Sweden", revenue: 87544.73, orders: 61, customers: 19 },
        { Country: "Denmark", revenue: 65842.25, orders: 36, customers: 12 }
    ],
    monthly_trend: [
        { YearMonth: "2009-12", revenue: 822483.95, orders: 1682, units: 425461 },
        { YearMonth: "2010-01", revenue: 651155.11, orders: 1105, units: 390672 },
        { YearMonth: "2010-02", revenue: 551504.72, orders: 1201, units: 381879 },
        { YearMonth: "2010-03", revenue: 840428.16, orders: 1664, units: 531309 },
        { YearMonth: "2010-04", revenue: 638647.98, orders: 1324, units: 379059 },
        { YearMonth: "2010-05", revenue: 760010.59, orders: 1547, units: 479155 },
        { YearMonth: "2010-06", revenue: 769530.85, orders: 1555, units: 461414 },
        { YearMonth: "2010-07", revenue: 694086.32, orders: 1461, units: 418195 },
        { YearMonth: "2010-08", revenue: 686705.47, orders: 1367, units: 446219 },
        { YearMonth: "2010-09", revenue: 947230.72, orders: 1785, units: 569766 },
        { YearMonth: "2010-10", revenue: 1165215.17, orders: 2191, units: 666792 },
        { YearMonth: "2010-11", revenue: 1471018.66, orders: 2835, units: 757827 },
        { YearMonth: "2010-12", revenue: 1083907.03, orders: 2004, units: 512217 },
        { YearMonth: "2011-01", revenue: 663365.74, orders: 1184, units: 396883 },
        { YearMonth: "2011-02", revenue: 537130.22, orders: 1162, units: 310565 },
        { YearMonth: "2011-03", revenue: 733853.53, orders: 1572, units: 424263 },
        { YearMonth: "2011-04", revenue: 533475.25, orders: 1290, units: 322442 },
        { YearMonth: "2011-05", revenue: 778007.82, orders: 1794, units: 466810 },
        { YearMonth: "2011-06", revenue: 770808.57, orders: 1656, units: 427926 },
        { YearMonth: "2011-07", revenue: 727521.90, orders: 1553, units: 429777 },
        { YearMonth: "2011-08", revenue: 760677.81, orders: 1500, units: 440263 },
        { YearMonth: "2011-09", revenue: 1045958.07, orders: 2006, units: 597148 },
        { YearMonth: "2011-10", revenue: 1146745.24, orders: 2176, units: 671239 },
        { YearMonth: "2011-11", revenue: 1515286.41, orders: 3004, units: 821379 },
        { YearMonth: "2011-12", revenue: 688880.85, orders: 909, units: 316448 }
    ],
    spend_distribution: {
        '0-100': 217,
        '100-500': 1849,
        '500-1K': 1107,
        '1K-5K': 2060,
        '5K-10K': 384,
        '10K-50K': 227,
        '50K+': 34
    },
    order_distribution: {
        '1': 1623,
        '2-3': 1608,
        '4-5': 846,
        '6-10': 925,
        '11-20': 579,
        '20+': 297
    }
};

// Customer Recommendation Profiles (Sample Catalogue)
const CUSTOMER_RECS = {
    18102: [
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.96 },
        { name: "WHITE HANGING HEART T-LIGHT HOLDER", code: "85123A", score: 0.92 },
        { name: "PARTY BUNTING", code: "47566", score: 0.88 },
        { name: "PAPER CHAIN KIT 50'S CHRISTMAS", code: "22086", score: 0.83 },
        { name: "ASSORTED COLOUR BIRD ORNAMENT", code: "84879", score: 0.79 }
    ],
    14646: [
        { name: "JUMBO BAG RED WHITE SPOTTY", code: "85099B", score: 0.95 },
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.91 },
        { name: "SPACEBOY LUNCH BOX", code: "22384", score: 0.87 },
        { name: "ROUND SNACK BOXES SET OF 4", code: "22720", score: 0.84 },
        { name: "ALARM CLOCK BAKELIKE GREEN", code: "22726", score: 0.78 }
    ],
    14156: [
        { name: "WHITE HANGING HEART T-LIGHT", code: "85123A", score: 0.94 },
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.90 },
        { name: "HEART OF WICKER SMALL", code: "84946", score: 0.86 },
        { name: "NATURAL SLATE HEART CHALKBOARD", code: "22457", score: 0.81 },
        { name: "WOODEN PICTURE FRAME WHITE FINISH", code: "21733", score: 0.75 }
    ],
    14911: [
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.94 },
        { name: "WHITE HANGING HEART T-LIGHT", code: "85123A", score: 0.89 },
        { name: "JUMBO BAG RED WHITE SPOTTY", code: "85099B", score: 0.82 },
        { name: "ASSORTED COLOUR BIRD ORNAMENT", code: "84879", score: 0.76 },
        { name: "PACK OF 72 RETROSPOT CAKE CASES", code: "22326", score: 0.71 }
    ],
    17450: [
        { name: "WHITE HANGING HEART T-LIGHT", code: "85123A", score: 0.93 },
        { name: "PARTY BUNTING", code: "47566", score: 0.87 },
        { name: "HEART OF WICKER LARGE", code: "84947", score: 0.82 },
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.79 },
        { name: "VINTAGE SNAP CARDS", code: "22730", score: 0.73 }
    ],
    default: [
        { name: "REGENCY CAKESTAND 3 TIER", code: "22423", score: 0.91 },
        { name: "WHITE HANGING HEART T-LIGHT", code: "85123A", score: 0.87 },
        { name: "JUMBO BAG RED WHITE SPOTTY", code: "85099B", score: 0.81 },
        { name: "PARTY BUNTING", code: "47566", score: 0.75 },
        { name: "ASSORTED COLOUR BIRD ORNAMENT", code: "84879", score: 0.70 }
    ]
};

// Currency Formatter
function formatGBP(num) {
    return '£' + Math.round(num).toLocaleString('en-GB');
}

function formatNumber(num) {
    return Math.round(num).toLocaleString('en-GB');
}

// ==========================================================================
// Particles Background Canvas
// ==========================================================================
function initParticles() {
    const canvas = document.getElementById('particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const numParticles = Math.min(width > 1200 ? 55 : 30, 70);
    const particles = [];

    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.45,
            vy: (Math.random() - 0.5) * 0.45,
            radius: Math.random() * 1.6 + 0.8,
            alpha: Math.random() * 0.4 + 0.2
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(99, 102, 241, ${p.alpha})`;
            ctx.fill();

            // Connect nearby particles
            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 130) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(168, 85, 247, ${0.15 * (1 - dist / 130)})`;
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
}

// ==========================================================================
// Animated Counter
// ==========================================================================
function initCounters() {
    const counters = document.querySelectorAll('.stat-number');
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-count'), 10);
        const duration = 2000;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(easeOut * target);

            counter.textContent = current.toLocaleString('en-GB');

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                counter.textContent = target.toLocaleString('en-GB');
            }
        }
        requestAnimationFrame(update);
    });
}

// ==========================================================================
// Navbar Scroll & Section Spy
// ==========================================================================
function initNavigation() {
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        let currentSectionId = '';
        sections.forEach(sec => {
            const top = sec.offsetTop - 120;
            const height = sec.offsetHeight;
            if (window.scrollY >= top && window.scrollY < top + height) {
                currentSectionId = sec.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSectionId}`) {
                link.classList.add('active');
            }
        });
    });
}

// ==========================================================================
// Chart.js Global Dark Defaults
// ==========================================================================
function setupChartDefaults() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.95)';
    Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
    Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(99, 102, 241, 0.3)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.displayColors = true;
}

// ==========================================================================
// Chart 1: Monthly Revenue & Orders Trend
// ==========================================================================
function renderRevenueChart() {
    const ctx = document.getElementById('revenueChart');
    if (!ctx) return;

    const labels = DATA.monthly_trend.map(m => m.YearMonth);
    const revenueData = DATA.monthly_trend.map(m => m.revenue);
    const ordersData = DATA.monthly_trend.map(m => m.orders);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Revenue (£)',
                    data: revenueData,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.12)',
                    fill: true,
                    tension: 0.38,
                    borderWidth: 2.5,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#818cf8',
                    yAxisID: 'y'
                },
                {
                    label: 'Orders',
                    data: ordersData,
                    borderColor: '#a855f7',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.38,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    pointBackgroundColor: '#c084fc',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return `Revenue: £${Math.round(context.raw).toLocaleString('en-GB')}`;
                            } else {
                                return `Orders: ${context.raw.toLocaleString('en-GB')}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        callback: val => '£' + (val / 1000).toFixed(0) + 'k'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: {
                        callback: val => val.toLocaleString('en-GB')
                    }
                }
            }
        }
    });
}

// ==========================================================================
// Chart 2: Revenue by Country
// ==========================================================================
function renderCountryChart() {
    const ctx = document.getElementById('countryChart');
    if (!ctx) return;

    const countries = DATA.top_countries.slice(0, 8);
    const labels = countries.map(c => c.Country);
    const revenues = countries.map(c => c.revenue);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Revenue (£)',
                data: revenues,
                backgroundColor: [
                    'rgba(99, 102, 241, 0.85)',
                    'rgba(168, 85, 247, 0.85)',
                    'rgba(6, 182, 212, 0.85)',
                    'rgba(16, 185, 129, 0.85)',
                    'rgba(245, 158, 11, 0.85)',
                    'rgba(244, 63, 94, 0.85)',
                    'rgba(59, 130, 246, 0.85)',
                    'rgba(147, 51, 234, 0.85)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Revenue: £${Math.round(ctx.raw).toLocaleString('en-GB')}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        callback: val => '£' + (val >= 1000000 ? (val / 1000000).toFixed(1) + 'M' : (val / 1000).toFixed(0) + 'k')
                    }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// ==========================================================================
// Chart 3: Customer Spending Distribution
// ==========================================================================
function renderSpendChart() {
    const ctx = document.getElementById('spendChart');
    if (!ctx) return;

    const labels = Object.keys(DATA.spend_distribution);
    const values = Object.values(DATA.spend_distribution);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Customers',
                data: values,
                backgroundColor: 'rgba(168, 85, 247, 0.75)',
                hoverBackgroundColor: 'rgba(168, 85, 247, 0.95)',
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.raw.toLocaleString('en-GB')} Customers`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Spend Bracket (£)', color: '#64748b', font: { size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Number of Customers', color: '#64748b', font: { size: 11 } }
                }
            }
        }
    });
}

// ==========================================================================
// Chart 4: Top 10 Products by Revenue
// ==========================================================================
function renderProductRevenueChart() {
    const ctx = document.getElementById('productRevenueChart');
    if (!ctx) return;

    const prods = DATA.top_products.slice(0, 8);
    const labels = prods.map(p => p.Description.length > 22 ? p.Description.substring(0, 22) + '...' : p.Description);
    const revenues = prods.map(p => p.TotalRevenue);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Revenue (£)',
                data: revenues,
                backgroundColor: 'rgba(6, 182, 212, 0.8)',
                hoverBackgroundColor: 'rgba(6, 182, 212, 1)',
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Revenue: £${Math.round(ctx.raw).toLocaleString('en-GB')}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { callback: val => '£' + (val / 1000).toFixed(0) + 'k' }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// ==========================================================================
// Chart 5: Order Frequency Distribution
// ==========================================================================
function renderOrderFreqChart() {
    const ctx = document.getElementById('orderFreqChart');
    if (!ctx) return;

    const labels = Object.keys(DATA.order_distribution).map(k => k + ' orders');
    const values = Object.values(DATA.order_distribution);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#6366f1',
                    '#8b5cf6',
                    '#06b6d4',
                    '#10b981',
                    '#f59e0b',
                    '#f43f5e'
                ],
                borderWidth: 2,
                borderColor: '#0f172a'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 }
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString('en-GB')} customers`
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// ==========================================================================
// Chart 6: Monthly Orders & Units Sold
// ==========================================================================
function renderOrdersChart() {
    const ctx = document.getElementById('ordersChart');
    if (!ctx) return;

    const labels = DATA.monthly_trend.map(m => m.YearMonth);
    const unitsData = DATA.monthly_trend.map(m => m.units);
    const ordersData = DATA.monthly_trend.map(m => m.orders);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'Units Sold',
                    data: unitsData,
                    backgroundColor: 'rgba(16, 185, 129, 0.65)',
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    type: 'line',
                    label: 'Order Count',
                    data: ordersData,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 2,
                    pointBackgroundColor: '#fbbf24',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 }
                },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            if (ctx.datasetIndex === 0) {
                                return `Units Sold: ${ctx.raw.toLocaleString('en-GB')}`;
                            } else {
                                return `Orders: ${ctx.raw.toLocaleString('en-GB')}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { maxRotation: 45, minRotation: 45, font: { size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { callback: val => (val / 1000).toFixed(0) + 'k' }
                },
                y1: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { callback: val => val.toLocaleString('en-GB') }
                }
            }
        }
    });
}

// ==========================================================================
// Populate Tables
// ==========================================================================
function populateTables() {
    // Customer Table
    const customerTable = document.getElementById('customerTable');
    if (customerTable) {
        customerTable.innerHTML = DATA.top_customers.map((c, idx) => `
            <tr style="cursor: pointer" onclick="selectCustomer(${c.CustomerID})">
                <td>#${idx + 1}</td>
                <td><strong>${c.CustomerID}</strong></td>
                <td>${formatGBP(c.TotalSpend)}</td>
                <td>${formatNumber(c.TotalOrders)}</td>
                <td>${formatNumber(c.TotalUnits)}</td>
                <td>${formatNumber(c.UniqueProducts)}</td>
                <td><span class="segment-badge high-value">${c.Segment}</span></td>
            </tr>
        `).join('');
    }

    // Product Table
    const productTable = document.getElementById('productTable');
    if (productTable) {
        productTable.innerHTML = DATA.top_products.map((p, idx) => `
            <tr>
                <td>#${idx + 1}</td>
                <td><strong>${p.StockCode}</strong></td>
                <td>${p.Description}</td>
                <td><strong>${formatGBP(p.TotalRevenue)}</strong></td>
                <td>${formatNumber(p.TotalQuantitySold)}</td>
                <td>${formatNumber(p.UniqueCustomers)}</td>
                <td>${formatNumber(p.TotalInvoices)}</td>
            </tr>
        `).join('');
    }
}

// ==========================================================================
// Customer Selection & Recommendations
// ==========================================================================
function selectCustomer(customerId) {
    const customer = DATA.top_customers.find(c => c.CustomerID == customerId);
    const profileName = document.getElementById('profileName');
    const profileSegment = document.getElementById('profileSegment');
    const pmSpend = document.getElementById('pmSpend');
    const pmOrders = document.getElementById('pmOrders');
    const pmUnits = document.getElementById('pmUnits');
    const pmProducts = document.getElementById('pmProducts');

    if (customer) {
        profileName.textContent = `Customer #${customer.CustomerID}`;
        profileSegment.textContent = customer.Segment;
        pmSpend.textContent = formatGBP(customer.TotalSpend);
        pmOrders.textContent = formatNumber(customer.TotalOrders);
        pmUnits.textContent = formatNumber(customer.TotalUnits);
        pmProducts.textContent = formatNumber(customer.UniqueProducts);
    } else {
        // Fallback for custom searched ID
        profileName.textContent = `Customer #${customerId}`;
        profileSegment.textContent = 'Active Client';
        pmSpend.textContent = '£12,450';
        pmOrders.textContent = '14';
        pmUnits.textContent = '2,890';
        pmProducts.textContent = '86';
    }

    // Update Recommendations UI
    updateRecommendations(customerId);

    // Scroll to recommendations or profile
    const recSection = document.getElementById('recommendations');
    if (recSection) {
        recSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function updateRecommendations(customerId) {
    const recs = CUSTOMER_RECS[customerId] || CUSTOMER_RECS.default;
    const recContainer = document.querySelector('.rec-cards');
    const recTitle = document.querySelector('.rec-customer h3');
    const recDesc = document.querySelector('.rec-desc');

    if (recTitle) {
        recTitle.textContent = `Recommendations for Customer #${customerId}`;
    }
    if (recDesc) {
        recDesc.textContent = `Ranked by Logistic Regression purchase probability across 500 candidate catalog items`;
    }

    if (recContainer) {
        recContainer.innerHTML = recs.map((r, i) => `
            <div class="rec-card" style="--delay: ${i}">
                <div class="rec-rank">#${i + 1}</div>
                <div class="rec-info">
                    <span class="rec-name">${r.name}</span>
                    <span class="rec-code">SKU: ${r.code}</span>
                </div>
                <div class="rec-score">
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${Math.round(r.score * 100)}%"></div>
                    </div>
                    <span class="score-value">${r.score.toFixed(2)}</span>
                </div>
            </div>
        `).join('');
    }
}

// Customer Search Input Handler
function initCustomerSearch() {
    const input = document.getElementById('customerSearch');
    if (!input) return;

    input.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') {
            const val = input.value.trim();
            if (val) {
                selectCustomer(val);
            }
        }
    });
}

// Make selectCustomer accessible globally for inline table clicks
window.selectCustomer = selectCustomer;

// ==========================================================================
// Initialization on DOM Loaded
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initCounters();
    initNavigation();
    setupChartDefaults();

    // Render Charts
    renderRevenueChart();
    renderCountryChart();
    renderSpendChart();
    renderProductRevenueChart();
    renderOrderFreqChart();
    renderOrdersChart();

    // Populate data tables
    populateTables();

    // Customer Search
    initCustomerSearch();
});
