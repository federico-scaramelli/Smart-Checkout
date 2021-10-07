% Function segmentImageEdge() 
% INPUT:  im, immagine da segmentare, etichettare, e tagliare.
% OUTPUT: Array contenente le immagini ritagliate e le maschere 
%         estratte da im.
% 
% La funzione ridimensiona inizialmente l'immagine per migliorare le
% performance e ridurre il rumore dovuto all'acquisizione. Se l'immagine è
% RGB, la converte in grayscale. Successivamente all'applicazione di un
% filtro mediano atto alla riduzione del rumore, vengono
% calcolati gli edge dell'immagine tramite l'algoritmo di Canny, con
% threshold calcolato con il medesimo algoritmo. Successivamente, tali edge
% vengono trasformati in maschere binarie complete grazie ad operazioni
% morfologiche matematiche.
% Viene poi effettuato il labeling delle regioni connesse, che permette di
% calcolare l'area e la bounding box di ognuna di esse. Gli oggetti con area
% <1/7 dell'area piu grande vengono scartati in quanto considerati errori
% di segmentazione. 
% Come ultima operazione la funzione applica ogni maschera dei vari oggetti
% all'immagine originale, salvando i risultati nell'array di celle `images`.
 

function [images, masks] = segmentImageEdge(im)
    im = imresize(im, 0.5); %Migliora le performance e riduce il rumore

    [~, ~, ch] = size(im);
    if(ch > 1)
        gray = rgb2gray(im);
    else
        error('È richiesto che l`immagine sia a colori.');
    end
    
    %Applico 3 filtri mediani per alleviare il rumore
    gray = medfilt2(medfilt2(medfilt2(gray)));
    
    %Estraggo gli edge tramite algoritmo di Canny con threshold automatico
    binaryEdge = edge(gray, 'canny');
    
    binary = imdilate(binaryEdge, strel('square', 3)); %Dilato un po gli edge 
    %Unisco eventuali punti separati che dovrebbero essere uniti
    binary = imclose(binary, strel('line', 5, 90));
    binary = imclose(binary, strel('line', 5, 0));
    binary = imfill(binary, 'holes');               %Riempio l'interno
    binary = imopen(binary, strel('square', 15));   %Elimino piccoli difetti 
    binary = imclearborder(binary, 8);              %Elimino regioni ai bordi
    binary = imerode(binary, strel('diamond', 1));  %Smusso la maschera
    
    %Labeling delle regioni connesse e calcolo delle proprietà di ognuna
    labeled = bwlabel(binary, 8);
    blobMeasurements = regionprops(binary,  'all');
    nBlobs = size(blobMeasurements, 1); % numero regioni connesse
  
    %Estraggo il bounding box e l'area dalle regionprops;
    areas = [blobMeasurements.Area];
    maxArea = max(areas);
    nObj=0;
    images = cell(nBlobs,1);
    masks = cell(nBlobs, 1);
    for i=1:1:nBlobs
        blobArea = areas(i);
        %Azzero le regioni con area <1/7 dell'area massima;
        if(blobArea < maxArea/7)
            blobPxls = blobMeasurements(i).PixelIdxList;
            labeled(blobPxls) = 0;
            continue;
        else 
            nObj = nObj+1; 
            blobBBox = blobMeasurements(i).BoundingBox; %BBox obj visitato
            cropped = im2double(imcrop(im, blobBBox));   %Crop img originale
            
            mask = imcrop((labeled==i), blobBBox); %Ridimensiono mask
            mask = mask~=0; %Trasformo in 1 eventuali etichette >1
            
            newImage =  mask .* cropped; 
            
            images{nObj} = newImage; 
            masks{nObj} = mask;
        end
    end 
    
    images = images(1:nObj); %Taglio i valori rimasti vuoti
    masks = masks(1:nObj);
end