# PDI_BPCS
Source code of original BPCS from Guilherme dos Santos Marcon NUSP 9293564

"Hiding images in images by BPCS method (Bit-Plane Complexity Segmentation)

Either aim of the project and intuitively implement the BPCS method of steganography, which in this case will serve to hide an image in another to be recovered. The method consists in hiding pixels from the image in blocks of a bit plane whose bits are, if they have noisy behavior, it takes advantage of the characteristic of human vision to concentrate not to reconcile patterns and shapes.

All the images used are non-directory "images" and the forms removed and modified from the site https://www.pexels.com/public-domain-images/, being transformed for the png format and reduced or size for simpler tests. The links of each individual image are saved in the ImagesLinks.txt file.

The main methods used are: read, save and manipulate the images using the imageio and numpy python libraries; transform imagem from Pure Binary Code to Canonical Gray Code and vice-versa; Check if a block of a bit plane is considered complex.

By Git do not receive files larger than 25MB, some repository images and two tests will not be available."

# Added GUI interface to make program more interactive and user friendly
Made use of Tkinter's side properties for proper placement

# Instructions
Run BPCS.py
- python BPCS.py

To embed a Target Image, ensure chosen Vessel Image is bigger than Target Image in terms of size and dimensions. 

User can select images from the images folder. Final image that has been embedded will be placed in main directory and named as "finalstego.png"

To recover a hidden Image, ensure "finalstego.png" is chosen. Hidden image will then be previewed on the GUI and printed out on the main directory as "HiddenImg.png"
