function imagenes_entrenamiento = unpack_dataset(file_name)
    % Abrir el archivo IDX de imágenes de entrenamiento
    fid = fopen(file_name, 'r', 'b'); % 'r' para lectura, 'b' para leer en modo binario

    % Leer los primeros 4 bytes (magic number)
    magic_number = fread(fid, 1, 'int32');

    % Leer el número de imágenes de entrenamiento
    num_imagenes = fread(fid, 1, 'int32');

    % Leer el número de filas y columnas de cada imagen
    num_filas = fread(fid, 1, 'int32');
    num_columnas = fread(fid, 1, 'int32');

    % Leer las imágenes de entrenamiento
    imagenes_entrenamiento = fread(fid, [num_filas*num_columnas, num_imagenes], 'uint8');

    % Cerrar el archivo IDX
    fclose(fid);

    % Redimensionar la matriz de imágenes de entrenamiento
    imagenes_entrenamiento = imagenes_entrenamiento';
    imagenes_entrenamiento = reshape(imagenes_entrenamiento, [num_imagenes, num_filas*num_columnas]);

    % Mostrar la forma (size) de la matriz de imágenes de entrenamiento
    disp(['Forma de la matriz de imágenes: ' num2str(size(imagenes_entrenamiento))]);

end