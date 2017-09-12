from PIL import Image

def main():
    try:
        img=Image.open("itachi.jpg")
        img=img.()
        img.save("itachi_cropped.png")
    except IOError:
        pass

if __name__=="__main__":
    main()
