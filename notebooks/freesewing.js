import Aaron from 'https://cdn.skypack.dev/@freesewing/aaron';
import theme from 'https://cdn.skypack.dev/@freesewing/plugin-theme';

const svg = new Aaron({ 
    sa: 10, // Seam allowance
    paperless: true, // Enable paperless mode
    // More settings, see: https://FreeSewing.dev/reference/api/settings
    measurements: { // Pass in measurements
      biceps: 387,
      chest: 1105,
      hips: 928,
      hpsToWaistBack: 502,
      neck: 420,
      shoulderSlope: 13,
      shoulderToShoulder: 481,
      waistToHips: 139,
    }
  })
  .use(theme) // Load theme plugin
  .draft() // Draft the pattern
  .render()

  document.getElementById('container').innerHTML = svg