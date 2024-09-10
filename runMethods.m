function [im, err, mg, ig] = runMethods(method,orig,imtotal,masksdrive1,resultspath,nof,Si,scales,nbSca,angles,nbAng)
if strcmp(method,'orig')
    im = orig;
    savetocsv(im,'imGCh',resultspath, nof);
    for i = 1:imtotal
        err(1,:,i) = performance_eval(orig(:,:,i),masksdrive1(:,:,i));
    end
    [mg,ig] = max(err(1,1,:));
    errp    = permute(err,[3,1,2]);
    savetocsv(errp,'DataGCh',resultspath,nof);
elseif strcmp(method,'emd')
    for i = 1:imtotal
        im(:,:,i) = emd(orig(:,:,i));
    end
    savetocsv(im,'imemd',resultspath, nof);
    for i = 1:imtotal
        err(1,:,i) = performance_eval(im(:,:,i),masksdrive1(:,:,i));
    end
    [mg,ig] = max(err(1,1,:));
    errp    = permute(err,[3,1,2]);
    savetocsv(errp,'DataEMD',resultspath,nof);
elseif strcmp(method,'vmd')
    for i = 1:imtotal
        im(:,:,i) = vmd(orig(:,:,i));
    end
    savetocsv(im,'imvmd',resultspath, nof);
    for i = 1:imtotal
        err(1,:,i) = performance_eval(im(:,:,i),masksdrive1(:,:,i));
    end
    [mg,ig] = max(err(1,1,:));
    errp    = permute(err,[3,1,2]);
    savetocsv(errp,'DataVMD',resultspath,nof);
elseif strcmp(method,'fmd')
    for i = 1:imtotal
        im(:,:,i) = fmd(orig(:,:,i));
    end
%     savetocsv(im,'imfmd',resultspath, nof);
    for i = 1:imtotal
        err(1,:,i) = performance_eval(im(:,:,i),masksdrive1(:,:,i));
            %disp(err(1,1,i));
    end
    [mg,ig] = max(err(1,1,:));

    errp    = permute(err,[3,1,2]);
    savetocsv(errp,'DataFMD',resultspath,nof);    
elseif strcmp(method,'clahe')
    for i = 1:imtotal
        im(:,:,i) = adapthisteq(orig(:,:,i));
    end
    savetocsv(im,'imclahe',resultspath, nof);
    for i = 1:imtotal
        err(1,:,i) = performance_eval(im(:,:,i),masksdrive1(:,:,i));
    end
    [mg,ig] = max(err(1,1,:));
    errp    = permute(err,[3,1,2]);
    savetocsv(errp,'DataCLAHE',resultspath,nof);
elseif strcmp(method,'gabor')
    for i = 1:imtotal
        im(:,:,:,i) = gabortrans(orig(:,:,i),Si,scales,nbSca,angles,nbAng);
    end
    savetocsv(im,'imgabor',resultspath, nof);
    for i = 1:imtotal
        for k = 1:nbSca
            err(1,:,k,i) = performance_eval(im(:,:,k,i),masksdrive1(:,:,i));
        end
        [maxg(i),idxg(i)] = max(err(1,1,:,i));
    end
    [mg,ig] = max(maxg);
    savetocsv(err,'DataGabor',resultspath,nof);
elseif strcmp(method,'haar')
    for i = 1:imtotal
        im(:,:,:,i) = haartrans(orig(:,:,i),Si,scales,nbSca);
    end
    savetocsv(im,'imhaar',resultspath, nof);
    for i = 1:imtotal
        for k = 1:nbSca
            err(1,:,k,i) = performance_eval(im(:,:,k,i),masksdrive1(:,:,i));
        end
        [maxg(i),idxg(i)] = max(err(1,1,:,i));
    end
    [mg,ig] = max(maxg);
    savetocsv(err,'DataHaar',resultspath,nof);
elseif strcmp(method,'cdf')
    for i = 1:imtotal
        jk          = cdf97trans(orig(:,:,i), scales, nbSca);
        im(:,:,:,i) = wavecdf97(jk, scales, nbSca);
    end
    savetocsv(im,'imcdf',resultspath, nof);
    for i = 1:imtotal
        for k = 1:nbSca
            err(1,:,k,i) = performance_eval(im(:,:,k,i),masksdrive1(:,:,i));
        end
        [maxg(i),idxg(i)] = max(err(1,1,:,i));
    end
    [mg,ig] = max(maxg);
    savetocsv(err,'dataCDF',resultspath,nof);
end
