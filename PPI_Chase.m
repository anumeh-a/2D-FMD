clc;
clear all;
close all;

%% CHASEDB1

scales  = 2.^(0:5);
nbSca   = length(scales);
angles  = 0:pi/12:pi-pi/12;
nbAng   = length(angles);

path = '\CHASEDB1\';
imextsn = ('.jpg');
mnextsn = ('.png');
imfiles = dir(fullfile(path,['*',imextsn]));
imtotal = numel(imfiles);
mnfiles = dir(fullfile(path,['*1st*',mnextsn]));
mntotal = numel(mnfiles);

for i = 1:imtotal
    fileaddress      = strcat(path,imfiles(i).name);
    file             = imread(fileaddress);
    Ig               = file(:, :, 2);             % Green channel    
    adahist1         = imcomplement(Ig);          % Inverted green channel
    se               = strel('ball', 5, 5);
    godisk           = adahist1 - imopen(adahist1, se);
    Si               = size(Ig);
    Ig               = godisk;
    Ig               = double(Ig);
    Ig(Ig>255)       = 255;
    imemd(:,:,i)     = emd(Ig);
    %imgabor(:,:,:,i) = gabortrans(Ig,Si,scales,nbSca,angles,nbAng);
    %imhaar(:,:,:,i)  = haartrans(Ig,Si,scales,nbSca);
    %jk               = cdf97trans(Ig, scales, nbSca);
    %imcdf (:,:,:,i)  = wavecdf97(jk, scales, nbSca);
    %imclahe(:,:,i)   = adapthisteq(Ig);
end

for i = 1:mntotal
    fileaddress = strcat(path,mnfiles(i).name);
    file = imread(fileaddress);
    maskschase(:,:,i) = file;
end