clc;
clear;
close all;
more off;
fid=fopen('train_mit.set');
nin=fscanf(fid,'%d',1);
tr_pats=fscanf(fid,'%d',1);
[A cnt] = fscanf(fid,'%f',[nin+1,Inf]);
cnt = cnt/(nin+1);
fclose(fid);
Ptr= A(1:nin,:);%x = input patterns
dtr= A(nin+1,:); %d = desired outputs
fid=fopen('valid_mit.set');
nin=fscanf(fid,'%d',1);
val_pats=fscanf(fid,'%d',1);
[A cnt] = fscanf(fid,'%f',[nin+1,Inf]);
cnt = cnt/(nin+1);
fclose(fid);
Pval= A(1:nin,:);%x = input patterns
dval= A(nin+1,:); %d = desired outputs
fid=fopen('test_mit.set');
nin=fscanf(fid,'%d',1);
tst_pats=fscanf(fid,'%d',1);
[A cnt] = fscanf(fid,'%f',[nin+1,Inf]);
cnt = cnt/(nin+1);
fclose(fid);
Ptest= A(1:nin,:);%x = input patterns
dtst= A(nin+1,:); %d = desired outputs
img=imread('mitilini.tif');%diavazei eikonas
[rows cols bands]=size(img);%epistrefei diastaseis eikonas
tmap=zeros(rows,cols); %matrix me rows kai columns
tmap=uint8(tmap);
e=0.2;

ppc=input('How many prototypes per class?[1,2,5,10]:')
init_method=input('o-random,1-from tr_set:');
%Prepare codebook matrix
szcbk=ppc*4;
W=zeros(nin,szcbk);
epochs=1;
L=[];
for i=1:ppc
    L=[L, [0:3]];
end
cvtr=zeros(1,10);  

cvval=zeros(1,10);

cvtest=zeros(1,10);     
   
     %initialize codebook
     %tic
     if init_method==0
         m=mean(Ptr,2);
         s=std(Ptr,[],2);
         W= 2*diag(s)*(rand(size(W))-0.5)+m;
     else
         b= ceil(rand*tr_pats);
         for cbk=1:szcbk
             while(L(cbk)~=dtr(b))
                 b=mod(b,tr_pats)+1;
             end
           W(:,cbk)=Ptr(:,b);
         end
     end
     %toc
     tic
     a0=0;
    tmax=epochs*tr_pats;
     %LFM
     for t=1:tmax
          a=a0/(1+(0.002*t));
          w=mod(t-1,tr_pats)+1;
         ed=vecnorm(W-Ptr(:,w));
         [~,c3]=min(ed);
         %L(C)
         if L(c3)~=dtr(w)
             W(:,c3)=W(:,c3)-a*(Ptr(:,w)-W(:,c3));
             
     for q=1:ppc*4
         if L(q)==dtr(w)
             edd=vecnorm(W(:,q)-Ptr(:,w));
            if edd<ed(c3)
             W(:,q)=W(:,q)+a*(Ptr(:,w)-W(:,q));
            end
         end
     end
      
         end  
           
  end
         %counts
         trcut=0;
         for jtr=1:tr_pats
             ccctr=W-Ptr(:,jtr);
             eddtr=vecnorm(ccctr);
             [~,c3tr]=min(eddtr);
             if dtr(jtr)==L(c3tr)
                 trcut=trcut+1;
             end
         end
         tstcut=0;
         for jtst=1:tst_pats
             ccctst=W-Ptr(:,jtst);
             eddtst=vecnorm(ccctst);
             [~,c3tst]=min(eddtst);
             if dtr(jtst)==L(c3tst)
                 tstcut=tstcut+1;
             end
         end
         valcut=0;
         for jval=1:val_pats
             cccval=W-Ptr(:,jval);
             eddval=vecnorm(cccval);
             [~,c3val]=min(eddval);
             if dtr(jval)==L(c3val)
                 valcut=valcut+1;
             end
         end         
        
         
         
cvtr(i)=trcut/tr_pats*100;
cvtest(i)=tstcut/tst_pats*100;
cvval(i)=valcut/val_pats*100;
for r=1:rows
    for c=1:cols %ka8e grammi kai ka8e stili
        x=squeeze(img(r,c,:));%gia ka8e keli tou matrix ta syndeei ola mazi
       edl1=vecnorm(W-double(x(:,1)));%double to x epeidi einai uint kai ta W einai double
            [~,winner]=min(edl1);%krataei nikiti
            tmap(r,c)=L(winner);%vazei to label tou nikiti stin kainourgia matrix pou arxikopoih8ike stin arxi
    end
end
imshow(tmap,[]);
toc