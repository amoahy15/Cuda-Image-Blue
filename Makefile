all: 
	nvcc -o blur blur.cu
	nvcc -o sharedBlur sharedBlur.cu
run: all
	./blur grumpy.bmp
	./blur cattest.bmp
	./blur Bitmoji.bmp
	./sharedBlur grumpy.bmp
	./sharedBlur cattest.bmp
	./sharedBlur Bitmoji.bmp
clean:
	rm -f blur 
	rm -f sharedBlur
	rm -f cattest_blurred.bmp
	rm -f cattest_shared_blurred.bmp
	rm -f grumpy_blurred.bmp
	rm -f grumpy_shared_blurred.bmp
	rm -f Bitmoji_blurred.bmp
	rm -f Bitmoji_shared_blurred.bmp
	rm -f time.csv
	rm -f shared_time.csv