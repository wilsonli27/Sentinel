<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SoilSecura | Global Tillage Assessment</title>
    
    <script src="https://cdn.tailwindcss.com"></script>
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        body { margin: 0; padding: 0; background-color: #111827; color: white; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        #map { height: 100vh; width: 100vw; z-index: 1; }
        
        /* Glassmorphism Dark Theme Dashboard */
        #dashboard {
            position: absolute;
            top: 2rem;
            right: -100%; /* Hidden by default for smooth slide-in */
            width: 400px;
            max-height: calc(100vh - 4rem);
            z-index: 1000;
            background: rgba(31, 41, 55, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            overflow-y: auto;
        }
        #dashboard.active { right: 2rem; }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 10px; }
    </style>
</head>
<body class="overflow-hidden">

    <div id="map"></div>

    <div id="dashboard" class="rounded-2xl shadow-2xl p-6 flex flex-col gap-6">
        
        <div class="flex justify-between items-center border-b border-gray-700 pb-4">
            <div>
                <h2 id="region-title" class="text-2xl font-bold text-white tracking-tight">Select Region</h2>
                <p id="region-coords" class="text-sm text-gray-400 mt-1">Awaiting satellite feed...</p>
            </div>
            <button onclick="closeDashboard()" class="text-gray-400 hover:text-white transition-colors p-2 rounded-full hover:bg-gray-700">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
        </div>

        <div>
            <h3 class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">ResNet-50 AI Predictions</h3>
            <div class="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <canvas id="tillageChart" height="200"></canvas>
            </div>
        </div>

        <div>
            <h3 class="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Policy Action Plan</h3>
            <div id="action-steps" class="flex flex-col gap-3">
                </div>
        </div>
        
    </div>

    <script>
        // --- 1. DATA: MOCKING YOUR MODEL OUTPUT ---
        // In a true production app, you would fetch this data from your Python backend:
        // fetch('http://localhost:8000/predict?lat=41&lng=-90').then(res => res.json())
        const regionData = {
            "North American Corn Belt": {
                coords: [41.0, -90.0],
                predictions: [10, 80, 10], // Cons, Conv, Trad
                steps: ["Subsidize no-till drill equipment rentals.", "Implement tax credits for off-season cover crops.", "Host community agronomy workshops."]
            },
            "Canadian Prairies": {
                coords: [51.0, -105.0],
                predictions: [60, 10, 30],
                steps: ["Maintain carbon credit payouts to reward good behavior.", "Monitor soil moisture retention levels.", "Expand successful pilot programs."]
            },
            "Indo-Gangetic Plain": {
                coords: [27.0, 80.0],
                predictions: [5, 40, 55],
                steps: ["Strictly enforce bans on post-harvest stubble burning.", "Distribute subsidized 'Happy Seeder' machines.", "Launch village-level educational outreach."]
            },
            "South American Cerrado": {
                coords: [-12.0, -50.0],
                predictions: [40, 50, 10],
                steps: ["Enforce dry-season soil armor mandates.", "Link agricultural loans to zero-deforestation compliance.", "Provide grants for precision agriculture transition."]
            },
            "Australian Wheatbelt": {
                coords: [-32.0, 118.0],
                predictions: [80, 5, 15],
                steps: ["Publish regional case studies on drought resilience.", "Optimize targeted fertilizer application.", "Establish the region as a global benchmark."]
            }
        };

        // --- 2. INITIALIZE MAP (Satellite Base) ---
        const map = L.map('map', { zoomControl: false }).setView([20, 0], 3);
        L.control.zoom({ position: 'bottomleft' }).addTo(map);

        // Add beautiful Esri Satellite Imagery
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri'
        }).addTo(map);

        // --- 3. CHART SETUP ---
        let chartInstance = null;
        const ctx = document.getElementById('tillageChart').getContext('2d');

        function updateChart(dataArray) {
            if (chartInstance) chartInstance.destroy();
            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Conservational', 'Conventional', 'Traditional'],
                    datasets: [{
                        data: dataArray,
                        backgroundColor: ['#10B981', '#F59E0B', '#EF4444'], // Emerald, Amber, Red
                        borderWidth: 0,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#D1D5DB', padding: 20, font: { family: 'inherit' } } }
                    },
                    cutout: '70%'
                }
            });
        }

        // --- 4. MAP INTERACTION LOGIC ---
        // Custom marker icon to fit the SaaS theme
        const customIcon = L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="background-color:#3B82F6; width:16px; height:16px; border-radius:50%; border:3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });

        // Add markers to the map
        for (const [name, data] of Object.entries(regionData)) {
            const marker = L.marker(data.coords, { icon: customIcon }).addTo(map);
            
            // Add a subtle circle around the region
            L.circle(data.coords, {
                color: '#3B82F6',
                fillColor: '#3B82F6',
                fillOpacity: 0.1,
                radius: 500000 // 500km radius visual
            }).addTo(map);

            marker.on('click', () => {
                // Zoom in smoothly
                map.flyTo(data.coords, 6, { duration: 1.5 });
                
                // Update Dashboard Data
                document.getElementById('region-title').innerText = name;
                document.getElementById('region-coords').innerText = `Lat: ${data.coords[0]} | Lng: ${data.coords[1]}`;
                updateChart(data.predictions);
                
                // Update Steps
                const stepsContainer = document.getElementById('action-steps');
                stepsContainer.innerHTML = ''; // Clear old steps
                data.steps.forEach((step, index) => {
                    stepsContainer.innerHTML += `
                        <div class="flex items-start bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-blue-500 transition-colors">
                            <div class="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm font-bold mt-0.5">
                                ${index + 1}
                            </div>
                            <p class="ml-3 text-sm text-gray-300 leading-snug">${step}</p>
                        </div>
                    `;
                });

                // Slide Dashboard In
                document.getElementById('dashboard').classList.add('active');
            });
        }

        // Close Dashboard function
        window.closeDashboard = function() {
            document.getElementById('dashboard').classList.remove('active');
            map.flyTo([20, 0], 3, { duration: 1.5 }); // Zoom back out to world view
        }
    </script>
</body>
</html>