## Post-Focus

Post Focus is a Python/PILLOW (hopefully at some point C++/OpenCV) project based off of the commonly used "Background Blur" or "Portrait Mode" effects on smartphone applications today. The goal in creating this project was to analyze and reconstruct how those programs work to create one that I have much more control over. When done I want to be able to automatically generate a depth map from an input image, have the ability to edit the depth map using a brush, and dynamically choose which object in the image to focus the scene on.There will also be the ability to change the way the lens interacts with the camera, such as anamophic blur kernels, adding thick "rings" to the edges of the circular kernel, having the blur kernels bend outward near the edges of the photo to mimic a super-wide aperture, chromatic aberration, and noise.

### To-Do List
##### 1. Realistic Blur using Depth Map
##### 2. Edit Depth Map with brush
##### 3. Choose object to focus on
##### 4. Anamorphic lens blur
##### 5. Blur "rings" (I think it's called onion-ring bokeh?)
##### 6. Distorted blur (I think this one's called swirly bokeh?)
##### 7. Other lens artifacts - chromatic aberration, noise, highlights