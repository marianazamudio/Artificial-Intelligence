function etiquetas = unpack_labels(file_name)
    % Abrir el archivo IDX de etiquetas de entrenamiento
    fid = fopen(file_name, 'r', 'b'); % 'r' para lectura, 'b' para leer en modo binario

    % Leer los primeros 4 bytes (magic number)
    magic_number = fread(fid, 1, 'int32');

    % Leer el número de etiquetas de entrenamiento
    num_etiquetas = fread(fid, 1, 'int32');

    % Leer las etiquetas de entrenamiento
    etiquetas = fread(fid, [num_etiquetas, 1], 'uint8');

    % Cerrar el archivo IDX
    fclose(fid);
        % Abrir el archivo IDX de etiquetas 
    fid = fopen(file_name, 'r', 'b'); % 'r' para lectura, 'b' para leer en modo binario

    % Leer los primeros 4 bytes (magic number)
    magic_number = fread(fid, 1, 'int32');

    % Leer el número de etiquetas 
    num_etiquetas = fread(fid, 1, 'int32');

    % Leer las etiquetas 
    etiquetas = fread(fid, [num_etiquetas, 1], 'uint8');

    % Cerrar el archivo IDX
    fclose(fid);
    
    tamano = size(etiquetas);
    % Mostrar la forma (size) del vector de etiquetas
    fprintf('Forma del vector de etiquetas: %d  %d\n', tamano(1), tamano(2));
    
    
end
    