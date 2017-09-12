from PIL import Image
filename=input("Please enter an image: ")
picture = Image.open(filename)
r,g,b = picture.getpixel( (0,0) )
print("Red: {0}, Green: {1}, Blue: {2}".format(r,g,b))