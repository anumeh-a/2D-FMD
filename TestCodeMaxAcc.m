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
resultspath = '\DriveTest\';
mkdir(resultspath);
refimg      = gunzip("im0077.ppm.gz");
refimg      = imread(char(refimg));
impath      = 'test\images\';
mnpath      = 'test\1st_manual\';
mkpath      = 'test\mask\';
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
for i = 11
    fileaddress = strcat(path,impath,imfiles(i).name);
    maskaddress = strcat(path,mkpath,mkfiles(i).name);
    file        = imread(fileaddress);
    mask        = imread(maskaddress);
    Img         = file(:, :, 2);             % Green channel 
    Si          = size(Img);
    Ig          = Img.*(mask/255);
    Igg         = imhistmatch(Ig,refimg(:,:,2));
    Igc         = adapthisteq(Igg);
    se          = strel('ball',5,5);
    sca = blockproc(Igc, [64 64], @(block) imcomplement(block.data) - imopen(imcomplement(block.data), se), 'Border', [4 4]);
    imed        = medfilt2(Igc, [25 25]);
    Ithin       = sca ;
    se          = strel('ball',12,12);
    adahist1    = adapthisteq(imcomplement(Ig)); 
    godisk      = adahist1 - imopen(adahist1, se);
    Ig          = godisk + uint8(Ithin) -imed/30 ;
    orig(:,:,i) = Ig.*(mask/255);
end
%% Ground Truth
for i = 1:mntotal
    fileaddress        = strcat(path,mnpath,mnfiles(i).name);
    file               = imread(fileaddress);
    masksdrive2(:,:,i) = file;
end
%  savetocsv(masksdrive2,'drive2',resultspath,1);
%% Methods
% [imorig, errorig, maxorig, idxorig]      = runMethods('orig',orig,imtotal,masksdrive2,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imemd, erremd, maxemd, idxemd]          = runMethods('emd',double(orig),imtotal,masksdrive2,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imvmd, errvmd, maxvmd, idxvmd]          = runMethods('vmd',orig,imtotal,masksdrive2,resultspath,1,Si,scales,nbSca,angles,nbAng);
[imfmd, errfmd, maxfmd, idxfmd]          = runMethods('fmd',orig,imtotal,masksdrive2,resultspath,1,Si,scales,nbSca,angles,nbAng);
% [imclahe, errclahe, maxclahe, idxclahe]  = runMethods('clahe',orig,imtotal,masksdrive2,resultspath,1,Si,scales,nbSca,angles,nbAng);
[imgabor, errcgabor, maxgabor, idxgabor] = runMethods('gabor',orig,imtotal,masksdrive2,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imhaar, errhaar, maxhaar, idxhaar]      = runMethods('haar',orig,imtotal,masksdrive2,resultspath,6,Si,scales,nbSca,angles,nbAng);
% [imcdf, errcdf, maxcdf, idxcdf]          = runMethods('cdf',orig,imtotal,masksdrive2,resultspath,6,Si,scales,nbSca,angles,nbAng);


for i = 1:imtotal
    ibw(:,:,i) = bwareaopen(double(imbinarize(imgabor(:,:,2,i)+imbinarize(imfmd(:,:,i)))),18);
    vv1(:,:,i) = performance_eval(double(ibw(:,:,i)), masksdrive2(:,:,i));
    ibw3(:,:,i) = bwmorph(ibw(:,:,i), 'spur', inf);
    vv3(:,:,i) = performance_eval(double(ibw3(:,:,i)), masksdrive2(:,:,i));
end
figure
[maxf1,idxf1] = max(vv1(1,1,:));
[maxf3,idxf3] = max(vv3(1,1,:));
% 
% [hr,hg,hb,RGB2,RGB2gt] = calcend(ibw3(:,:,idxf3),masksdrive2(:,:,idxf3));
% imshow(RGB2);
% figure
% imshow(RGB2gt);