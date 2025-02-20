###https://www.hackerearth.com/practice/notes/extracting-pixel-values-of-an-image-in-python/

from PIL import Image

im = Image.open('try.png', 'r')

#print(im.size)
#print('width=214 pixels, height=530 pixels')

pix_val = list(im.getdata())

counter=0
for j in range(530):
    for i in range(214):
        R=pix_val[counter][0]
        G=pix_val[counter][1]
        B=pix_val[counter][2]
        val=0.30*R + 0.59*G + 0.11*B
        #print(214-i,530-j,R,G,B,val)
        counter+=1


im = Image.open('scale.png', 'r')
width=im.size[0]
height=im.size[1]
print(im.size,width,height)
pix_val = list(im.getdata())

counter=0
for j in range(height):
    for i in range(width):
        R=pix_val[counter][0]
        G=pix_val[counter][1]
        B=pix_val[counter][2]
        val=0.30*R + 0.59*G + 0.11*B
        print(width-i,height-j,R,G,B,val)
        counter+=1






#pix_val_flat = [x for sets in pix_val for x in sets]
#print(pix_val)

mine_file=open('mineRGB.xml',"w")
mine_file.write('<ColorMaps> \n')
mine_file.write('<ColorMap name="roma" space="HSV"> \n')

mine_file.write('</ColorMap> \n')
mine_file.write('</ColorMaps>')

#mine_file.write("%e %e %e \n" %())


mine_file.close()



