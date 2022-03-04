import streamlit as st
import streamlit.components.v1 as components

components.html("""
    hello world
    <div id="container" style="width: 450px; height: 600px">
      coco
    </div>

    <script type="module">
    import Aaron from 'https://cdn.skypack.dev/@freesewing/aaron';
    import theme from 'https://cdn.skypack.dev/@freesewing/plugin-theme';
    import svgAttr from 'https://cdn.skypack.dev/@freesewing/plugin-svgattr';

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
    .use(theme)
    .use(svgAttr, {width: "100%", height: "100%"})
    .draft() // Draft the pattern
    .render()

    document.getElementById('container').innerHTML = svg
  </script>""", height=600)
