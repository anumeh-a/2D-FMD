function savetocsv(file,name,resultspath,nof)


if size(file,4)>1
    disp('Multiple files being generated.');
    for i = 1:nof
        filename = [resultspath,name,num2str(i),'.csv'];
        x(:,:,:) = file(:,:,i,:);
        writematrix(x,filename);
    end
else
    disp('Single file being generated.');
    filename = [resultspath,name,'.csv'];
    x(:,:,:) = file(:,:,:);
    writematrix(x,filename);
end
