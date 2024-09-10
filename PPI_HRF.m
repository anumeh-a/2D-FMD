clc;
clear all;
close all;

% HRF

scales  = 2.^(0:5);
nbSca   = length(scales);
angles  = 0:pi/12:pi-pi/12;
nbAng   = length(angles);

path    = '\HRF\';
resultspath = '\csvdata\HRF\';
impath  = 'images\';
mnpath  = 'manual1\';
mkpath  = 'mask\';
imextsn = ('.jpg');
mnextsn = ('.tif');
mkextsn = ('.tif');
imfiles = dir(fullfile([path,impath],['*',imextsn]));
imtotal = numel(imfiles);
mnfiles = dir(fullfile([path,mnpath],['*',mnextsn]));
mntotal = numel(mnfiles);
mkfiles = dir(fullfile([path,mkpath],['*',mkextsn]));
mktotal = numel(mkfiles);

for i = 1:2
    fileaddress      = strcat(path,impath,imfiles(i).name);
    maskaddress      = strcat(path,mkpath,mkfiles(i).name);
    file             = imread(fileaddress);
    mask             = imread(maskaddress);
    Ig               = file(:, :, 2);             % Green channel
    mask             = mask(:,:,2);
    adahist1         = imcomplement(Ig);          % Inverted green channel
    se               = strel('ball', 5, 5);
    godisk           = adahist1 - imopen(adahist1, se);
    Si               = size(Ig);
    Ig               = godisk;
    Ig               = double(Ig);
    mask             = double(imbinarize(mask));
    Ig               = Ig.*mask;
    Ig(Ig>255)       = 255;
    orig(:,:,i)      = Ig;
end
%% Ground Truth
for i = 1:2
    fileaddress = strcat(path,mnpath,mnfiles(i).name);
    file = imread(fileaddress);
    maskshrf(:,:,i) = file;
end
% savetocsv(maskshrf,'maskshrf',resultspath,1);
%% Methods
[imorig, errorig, maxorig, idxorig]      = runMethods('orig',orig,imtotal,maskshrf,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imemd, erremd, maxemd, idxemd]          = runMethods('emd',orig,imtotal,maskshrf,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imvmd, errvmd, maxvmd, idxvmd]          = runMethods('vmd',orig,imtotal,maskshrf,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imclahe, errclahe, maxclahe, idxclahe]  = runMethods('clahe',orig,imtotal,maskshrf,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imgabor, errcgabor, maxgabor, idxgabor] = runMethods('gabor',orig,imtotal,maskshrf,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imhaar, errhaar, maxhaar, idxhaar]      = runMethods('haar',orig,imtotal,maskshrf,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imcdf, errcdf, maxcdf, idxcdf]          = runMethods('cdf',orig,imtotal,maskshrf,resultspath,6,Si,scales,nbSca,angles,nbAng);



