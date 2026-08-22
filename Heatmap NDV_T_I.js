// Saskatchewan Farmland NDVI Visualization
// 75km x 75km area - Peak Harvest Season 2023

// Initialize Google Earth Engine
// Set your project ID
var projectId = 'gvrsf-2026';

// Define area of interest (AOI) - 75km x 75km in Saskatchewan farmland region
// Coordinates: Central Saskatchewan agricultural area near Saskatoon
var centerLon = -106.647;
var centerLat = 52.147;

var aoi = ee.Geometry.Rectangle([
  centerLon - 0.525, // approximately 37.5km west
  centerLon + 0.525, // approximately 37.5km east
  centerLat - 0.3375, // approximately 37.5km south
  centerLat + 0.3375  // approximately 37.5km north
]);

// Define time period for 2023 peak harvest season (late July to mid-August)
// This is when crops are at maximum vegetation before harvest
var startDate = '2023-07-20';
var endDate = '2023-08-20';

// Load Sentinel-2 imagery
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2', 'B3', 'B4', 'B8']);

// Create median composite
var composite = s2Collection.median().clip(aoi);

// Calculate NDVI
// NDVI = (NIR - Red) / (NIR + Red)
// Using Sentinel-2: NIR = B8, Red = B4
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Visualization parameters
var ndviVis = {
  min: -0.2,
  max: 0.8,
  palette: ['brown', 'yellow', 'lightgreen', 'darkgreen']
};

// Center the map on AOI
Map.centerObject(aoi, 10);

// Add layers to map
Map.addLayer(aoi, {color: 'white'}, 'Area of Interest (75km x 75km)', false);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'RGB Composite');
Map.addLayer(ndvi, ndviVis, 'NDVI - Peak Harvest Season');

// Add legend for NDVI
var ndviLegend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var ndviTitle = ui.Label({
  value: 'NDVI (Peak Harvest)',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0'}
});
ndviLegend.add(ndviTitle);

var legendColorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: {
    bbox: [0, 0, 1, 0.1],
    dimensions: '200x20',
    format: 'png',
    min: 0,
    max: 1,
    palette: ndviVis.palette
  },
  style: {margin: '0 0 4px 0'}
});
ndviLegend.add(legendColorBar);

var legendLabels = ui.Panel({
  widgets: [
    ui.Label('-0.2 (Low)', {margin: '4px 0', fontSize: '12px'}),
    ui.Label('0.8 (High)', {margin: '4px 0', fontSize: '12px', textAlign: 'right'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});
ndviLegend.add(legendLabels);

var legendDescription = ui.Label({
  value: 'High values = healthy, mature crops',
  style: {fontSize: '11px', color: '666', margin: '4px 0 0 0'}
});
ndviLegend.add(legendDescription);

Map.add(ndviLegend);

// Print statistics
print('=== NDVI Statistics (Peak Harvest Season) ===');
print('Date Range:', startDate, 'to', endDate);
print('NDVI Values:', ndvi.reduceRegion({
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.minMax(),
    sharedInputs: true
  }),
  geometry: aoi,
  scale: 10,
  maxPixels: 1e9
}));

print('Number of images used:', s2Collection.size());

// Export option (uncomment to use)
/*
Export.image.toDrive({
  image: ndvi,
  description: 'Saskatchewan_NDVI_PeakHarvest_2023',
  folder: 'GEE_Exports',
  region: aoi,
  scale: 10,
  crs: 'EPSG:4326'
});
*/

print('NDVI visualization complete! Toggle layers in the Layers panel.');