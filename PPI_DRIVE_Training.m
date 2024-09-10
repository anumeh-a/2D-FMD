clc;
clear all;
close all;
addpath("support_functions");
%% DRIVE Training
%% Initiation
scales      = 2.^(0:5);
nbSca       = length(scales);
angles      = 0:pi/12:pi-pi/12;
nbAng       = length(angles);
path        = '\DRIVE\';
resultspath = '\DriveTraining\';
impath      = 'training\images\';
mnpath      = 'training\1st_manual\';
mkpath      = 'training\mask\';
imextsn     = ('.tif');
mnextsn     = ('.gif');
mkextsn     = ('.gif');
imfiles     = dir(fullfile([path,impath],['*',imextsn]));
imtotal     = numel(imfiles);
mnfiles     = dir(fullfile([path,mnpath],['*',mnextsn]));
mntotal     = numel(mnfiles);
mkfiles     = dir(fullfile([path,mkpath],['*',mkextsn]));
mktotal     = numel(mkfiles);
%% Image
for i = 1:imtotal
    fileaddress = strcat(path,impath,imfiles(i).name);
    maskaddress = strcat(path,mkpath,mkfiles(i).name);
    file        = imread(fileaddress);
    mask        = imread(maskaddress);
    Ig          = file(:, :, 2);             % Green channel    
    adahist1    = imcomplement(Ig);          % Inverted green channel
    se          = strel('ball', 5, 5);
    godisk      = adahist1 - imopen(adahist1, se);
    Si          = size(Ig);
    Ig          = godisk;
    Ig          = double(Ig);
    mask        = double(imbinarize(mask));
    Ig          = Ig.*mask;
    Ig(Ig>255)  = 255;
    orig(:,:,i) = Ig;
end
%% Ground Truth
for i = 1:mntotal
    fileaddress        = strcat(path,mnpath,mnfiles(i).name);
    file               = imread(fileaddress);
    masksdrive1(:,:,i) = file;
end
% savetocsv(masksdrive1,'masksdrive1',resultspath,1);
%% Methods
% [imorig, errorig, maxorig, idxorig]      = runMethods('orig',orig,imtotal,masksdrive1,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imemd, erremd, maxemd, idxemd]          = runMethods('emd',orig,imtotal,masksdrive1,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imvmd, errvmd, maxvmd, idxvmd]          = runMethods('vmd',orig,imtotal,masksdrive1,resultspath,1,Si,scales,nbSca,angles,nbAng);
[imfmd, errfmd, maxfmd, idxfmd]          = runMethods('fmd',orig,imtotal,masksdrive1,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imclahe, errclahe, maxclahe, idxclahe]  = runMethods('clahe',orig,imtotal,masksdrive1,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imgabor, errcgabor, maxgabor, idxgabor] = runMethods('gabor',orig,imtotal,masksdrive1,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imhaar, errhaar, maxhaar, idxhaar]      = runMethods('haar',orig,imtotal,masksdrive1,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imcdf, errcdf, maxcdf, idxcdf]          = runMethods('cdf',orig,imtotal,masksdrive1,resultspath,6,Si,scales,nbSca,angles,nbAng);
% for i = 1:imtotal
%     sty(:,:,i) = bwareaopen(imbinarize(imfmd(:,:,i)),10);
%     sty2(:,:,i) = bwareaopen(imbinarize(imemd(:,:,i)),100);
%     nr(:,:,i) = performance_eval(double(sty(:,:,i)),masksdrive1(:,:,i));
%     nr2(:,:,i) = performance_eval(double(sty2(:,:,i)),masksdrive1(:,:,i));
% end
% [maxnr,idxnr] = max(nr(1,1,:));
% [maxnr2,idxnr2] = max(nr2(1,1,:));
% newimg = imclose(imbinarize(imfmd(:,:,1)), strel('sphere', 2));
% % newimg = imbinarize(imfmd(:,:,1));
% % net = denoisingNetwork('DnCNN');
% % denoisedI = denoiseImage(double(newimg),net);
% subplot(1,3,1);
% imshow(masksdrive1(:,:,idxfmd));
% subplot(1,3,2);
% imshow(bwareaopen(double(imbinarize(imemd(:,:,idxfmd))),100));
% subplot(1,3,3);
% imshow(imbinarize(imfmd(:,:,idxfmd)));
for i = 1:imtotal
vv1(:,:,i) = performance_eval(orig(:,:,i), masksdrive1(:,:,i));
end
[maxf2,idxf2] = max(vv1(1,1,:))
