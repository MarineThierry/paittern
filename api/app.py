import streamlit as st
import streamlit.components.v1 as components
import cloudinary
import cloudinary.uploader
import os
from os.path import join,dirname
from dotenv import load_dotenv, find_dotenv

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


# load the video input into cloudinary



# point to .env file
env_path = join(dirname(dirname('__file__')),'.env') # ../.env
env_path = find_dotenv() # automatic find

# load your api key as environment variables
load_dotenv(env_path)

#config cloudinary authentification
cloudinary.config(
  cloud_name = "paittern",
  api_key = os.getenv('CLOUDINARY_API_KEY'),
  api_secret = os.getenv('CLOUDINARY_API_SECRET'),
  secure = True
)

#upload on cloud
response =\
cloudinary.uploader.unsigned_upload(
    "...",#add uploaded video from streamlit
    resource_type='video',
    upload_preset='paittern',
    public_id = "test")

url_video_path = response['url']
