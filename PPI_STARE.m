clc;
clear all;
close all;

% STARE

scales  = 2.^(0:5);
nbSca   = length(scales);
angles  = 0:pi/12:pi-pi/12;
nbAng   = length(angles);

path        = '\STARE\';
resultspath = '\csvdata\Stare\';
impath      = 'stare-images\';
mnpath      = 'labels-ah\';
imextsn     = ('.gz');
mnextsn     = ('.gz');
imfiles     = dir(fullfile([path,impath],['*',imextsn]));
imtotal     = numel(imfiles);
mnfiles     = dir(fullfile([path,mnpath],['*',mnextsn]));
mntotal     = numel(mnfiles);

for i = 1:imtotal
    fileaddress = strcat(path,impath,imfiles(i).name);
    fileppm     = gunzip(fileaddress);
    file        = imread(char(fileppm));
    Ig          = file(:, :, 2);             % Green channel
    Ig(Ig<50)   = 0;
    adahist1    = imcomplement(Ig);          % Inverted green channel
    se          = strel('ball', 5, 5);
    godisk      = adahist1 - imopen(adahist1, se);
    Ig          = godisk;
    Si          = size(Ig);
    Ig          = double(Ig);
    Ig(Ig>255)  = 255;
    orig(:,:,i) = Ig;
end

for i = 1:mntotal
    fileaddress = strcat(path,mnpath,mnfiles(i).name);
    fileppm = gunzip(fileaddress);
    file = imread(char(fileppm));
    masksstare(:,:,i) = file;
end
% % savetocsv(masksstare,'masks',resultspath,1);
% %% Methods
% % [imorig, errorig, maxorig, idxorig]      = runMethods('orig',orig,imtotal,masksstare,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imemd, erremd, maxemd, idxemd]          = runMethods('emd',orig,imtotal,masksstare,resultspath,1,Si,scales,nbSca,angles,nbAng);
% % [imvmd, errvmd, maxvmd, idxvmd]          = runMethods('vmd',orig,imtotal,masksstare,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imfmd, errfmd, maxfmd, idxfmd]          = runMethods('fmd',orig,imtotal,masksstare,resultspath,1,Si,scales,nbSca,angles,nbAng);
% % [imclahe, errclahe, maxclahe, idxclahe]  = runMethods('clahe',orig,imtotal,masksstare,resultspath,1,Si,scales,nbSca,angles,nbAng);
% % [imgabor, errcgabor, maxgabor, idxgabor] = runMethods('gabor',orig,imtotal,masksstare,resultspath,6,Si,scales,nbSca,angles,nbAng);
% % [imhaar, errhaar, maxhaar, idxhaar]      = runMethods('haar',orig,imtotal,masksstare,resultspath,6,Si,scales,nbSca,angles,nbAng);
% % [imcdf, errcdf, maxcdf, idxcdf]          = runMethods('cdf',orig,imtotal,masksstare,resultspath,6,Si,scales,nbSca,angles,nbAng);
% 
% newimg = imclose(imbinarize(imfmd(:,:,1)), strel('sphere', 2));
% % newimg = imbinarize(imfmd(:,:,1));
% % net = denoisingNetwork('DnCNN');
% % denoisedI = denoiseImage(double(newimg),net);
% subplot(1,3,1);
% imshow(masksstare(:,:,1));
% subplot(1,3,2);
% imshow(bwareaopen(double(imbinarize(imemd(:,:,1))),100));
% subplot(1,3,3);
% imshow(newimg);
% %% Resizing
% % imorig     = imresize(imorig,[605 605]);
% % savetocsv(imorig,'reimGCh',resultspath,1);
% % masksstare = imresize(masksstare,[605 605]);
% % savetocsv(masksstare,'remasks',resultspath,1);
% % imemd      = imresize(imemd,[605 605]);
% % savetocsv(imemd,'reimemd',resultspath,1);
% % imvmd      = imresize(imvmd,[605 605]);
% % savetocsv(imvmd,'reimvmd',resultspath,1);
% % imclahe    = imresize(imclahe,[605 605]);
% % savetocsv(imclahe,'reimclahe',resultspath,1);
% % imgabor    = imresize(imgabor,[605 605]);
% % savetocsv(imemd,'reimgabor',resultspath,6);
% % imhaar     = imresize(imhaar,[605 605]);
% % savetocsv(imhaar,'reimhaar',resultspath,6);
% imcdf      = imresize(imcdf,[605 605]);
% savetocsv(imcdf,'reimcdf',resultspath,6);

for i = 1:imtotal
vv1(:,:,i) = performance_eval(orig(:,:,i), masksstare(:,:,i));
end
[maxf2,idxf2] = max(vv1(1,1,:))