import streamlit as st
import streamlit.components.v1 as components

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Upload Your Video", type=["mp4", "MOV"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)


number = st.number_input('Enter your size (Height) in cm :')

st.write('You measure : ', number * 100, ' cm')

import streamlit as st
import tensorflow as tf




option = st.selectbox(
     'Choose your pAIttern !',
     ("Aaron", "Albert", "Bee", "Bella", "Benjamin","Bent", "Breanna", "Brian", "Bruce", "Carlita",
                 "Carlton", "Cathrin", "Charlie", "Cornelius", "Diana", "Florence", "Florent", "Holmes", "Hortensia", 
                 "Huey", "Hugo", "Jaeger", "Lunetius", "Paco", "Penelope", "Sandy", "Shin", "Simon", "Simone",
                 "Sven", "Tamiko", "Teagan", "Theo", "Tiberius", "Titan", "Trayvon", "Ursula",
                 "Wahid", "Walburga", "Waralee", "Yuri"))


st.write('You selected:', option)

from PIL import Image
image = Image.open(f'/home/jeremie/code/MarineThierry/paittern/paittern images/{option}.png')

st.image(image, caption=f'{option}')

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    bytes_data = img_file_buffer.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3)

    # Check the type of img_tensor:
    # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
    st.write(type(img_tensor))

    # Check the shape of img_tensor:
    # Should output shape: (height, width, channels)
    st.write(img_tensor.shape)




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
