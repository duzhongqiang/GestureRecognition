%% ����ʶ��ATMͼ
close all;
clear;
clc;

%% ��ʼ��
N = 256;
M = 8;
Frames = 32;  %֡��

% sss = {'houyi','latui','qiantui','tuila','tuiyou','tuizuo','youhua','youzuohua','zuohua','zuoyouhua'};
sss = {'TestData'};
for ss = sss
    path = char(ss);
    %% ·������
%     path = ['test'];
    list = dir(['G:\',path ,'\','*.mat']);
    k = length(list);
    path2 = ['D:\workplace\chongqing\GestureRecognition\DataSet\testData\',path ];
%     mkdir(path2);

    %% ����
    for i =1:k  %��k������
        str = strcat('G:\',path,'\',list(i).name);
        load(str);
        DataSize = size(adcData);   %���ݴ�С
        numsf = DataSize(2)/Frames; %ÿ֡�Ĳ�������
        numst = 256;                %��ʱ�����
        atm =[];  
        path3 = [path2,'\',num2str(i-1, '%04d')];
%         mkdir(path3);

        for m =1:1:Frames           %��ÿһ֡���ݽ��д���,��32֡
            adcDataFrame = adcData(1:8,(m-1)*numsf+1:m*numsf); %ÿһ֡�Ĳ�������
            X=adcDataFrame;

            mtiData = zeros(8,(64-1)*256);
            %MTI��������
            for ii=1:(64-1)*256     
                mtiData(:,ii)=adcDataFrame(:,ii+256)-adcDataFrame(:,ii);
            end
            subData=mtiData(:,32*256+1:(32+1)*256); %��64prt��ѡ��һ��prt

            Rx = subData*subData'/N;%��ؾ���
            %%MUSIC�㷨
            [uRx,lamdaRx] = eig(Rx);        %����ֽ�
            lamdax = diag(real(lamdaRx));   %����ֵ�ԽǾ���
            [slamda,nID] = sort(lamdax);    %����ֵ����
            G = uRx(:,nID(1:M-1));          %�����ӿռ�
            Ns = 100; 
            w = linspace(-pi/2,pi/2,Ns); 
            Pmusicw = zeros(1,Ns); %�׷�����
            for m = 1:Ns 
                Pmusicw(m)= 1./sum((abs(exp(1j*pi*sin(w(m))*(0:M-1))*G)).^2); 
            end 
            Pmusicw = (abs(Pmusicw)-min(abs(Pmusicw)))./ (max(abs(Pmusicw))-min(abs(Pmusicw)));
            atm=[atm Pmusicw'];
        end
        atm=imresize(atm,[224,224]);
        path4 = [path3,'\',num2str(32, '%02d'),'.png'];
        imwrite(abs(atm),path4);
        disp(['The ',num2str(i),' is finished!']);
    end
end