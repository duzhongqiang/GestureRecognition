%% 手势识别ATM图
close all;
clear;
clc;

%% 初始化
N = 256;
M = 8;
Frames = 32;  %帧数

sss = {'houyi','latui','qiantui','tuila','tuiyou','tuizuo','youhua','youzuohua','zuohua','zuoyouhua'};
for ss = sss
    %% 路径解析
    path = char(ss);

    list = dir(['D:\workplace\chongqing\GestureRecognition\DataSet\test2\matDataSet\',path ,'\','*.mat']);
    k = length(list);
    path2 = ['D:\workplace\chongqing\GestureRecognition\DataSet\test2\atmDataSet\',path ];
    mkdir(path2);

    %% 计算
    for i =1:k  %共k个数据
        str = strcat('D:\workplace\chongqing\GestureRecognition\DataSet\test2\matDataSet\',path,'\',list(i).name);
        load(str);
        DataSize = size(adcData);   %数据大小
        numsf = DataSize(2)/Frames; %每帧的采样点数
        numst = 256;                %快时间点数
        atm =[];  
        for m =1:1:Frames           %对每一帧数据进行处理,共32帧
            adcDataFrame = adcData(1:8,(m-1)*numsf+1:m*numsf); %每一帧的采样点数
            X=adcDataFrame;

            mtiData = zeros(8,(64-1)*256);
            %MTI滑动对消
            for ii=1:(64-1)*256     
                mtiData(:,ii)=adcDataFrame(:,ii+256)-adcDataFrame(:,ii);
            end
            subData=mtiData(:,32*256+1:(32+1)*256); %再64prt中选择一个prt

            Rx = subData*subData'/N;%相关矩阵
            %%MUSIC算法
            [uRx,lamdaRx] = eig(Rx);        %矩阵分解
            lamdax = diag(real(lamdaRx));   %特征值对角矩阵
            [slamda,nID] = sort(lamdax);    %特征值排列
            G = uRx(:,nID(1:M-1));          %噪声子空间
            Ns = 100; 
            w = linspace(-pi/2,pi/2,Ns); 
            Pmusicw = zeros(1,Ns); %谱峰搜索
            for m = 1:Ns 
                Pmusicw(m)= 1./sum((abs(exp(1j*pi*sin(w(m))*(0:M-1))*G)).^2); 
            end 
            Pmusicw = (abs(Pmusicw)-min(abs(Pmusicw)))./ (max(abs(Pmusicw))-min(abs(Pmusicw)));
            atm =[atm Pmusicw'];
        end
    %     atm =imresize(atm,[224,224]);
    %     maxp = max(max(abs(atm)));
    %     minp = min(min(abs(atm)));
    %     atm = atm/(maxp-minp);
    %     path3 = [path2,'\',num2str(i-1, '%04d'),'.png'];
    %     imwrite(abs(atm),path3);
        save([path2,'\',num2str(i-1, '%04d'),'.mat'],'atm');
        disp(['The ',num2str(i),' is finished!']);
    end
end