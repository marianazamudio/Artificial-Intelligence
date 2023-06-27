function images = normalize_images()
    images = [];
    image_paths = {'images/i0.jpg','images/i1.jpg', 'images/i2.jpg',  'images/i3.jpg',  'images/i4.jpg','images/i5.jpg', 'images/i6.jpg',  'images/i7.jpg',  'images/i8.jpg','images/i9.jpg', 'images/i0_1.jpg','images/i1_1.jpg', 'images/i2_1.jpg',  'images/i3_1.jpg',  'images/i4_1.jpg','images/i5_1.jpg','images/i6_1.jpg', 'images/i7_1.jpg',  'images/i8_1.jpg',  'images/i9_1.jpg'} ;

    for i = 1:length(image_paths)
        path = image_paths{i};
        norm_image = normalize_image(path);
        images(end+1,:) = norm_image();
    end
    
    
    
end
