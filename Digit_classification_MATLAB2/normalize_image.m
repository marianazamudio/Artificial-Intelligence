function pixel_list = normalize_image(image_path)
    % Leer la imagen
    image = imread(image_path);

    % Redimensionar la imagen a 28x28 píxeles
    resized_image = imresize(image, [28, 28]);

    % Convertir la imagen a escala de grises
    gray = rgb2gray(resized_image);
    % Espejear 
    matriz_espejo = fliplr(gray);
    % Girar
    matriz_girada = rot90(matriz_espejo, 1);
    
    matriz_girada(matriz_girada ==3)= 0;
    % Normalizar los valores de píxeles entre 0 y 1
    normalized = double(matriz_girada) / 255.0;
    
    % Aplanar la imagen en una lista unidimensional
    pixel_list = reshape(normalized, 1, []);
end


