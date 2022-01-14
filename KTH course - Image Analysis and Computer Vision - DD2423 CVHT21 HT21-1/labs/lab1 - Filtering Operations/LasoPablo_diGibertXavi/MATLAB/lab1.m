Fhat=zeros(128,128);
p=5;
q=9;
Fhat(p,q)=1;
F=ifft2(Fhat);
Fabsmax = max(abs(F(:)));
figure(1);
showgrey(real(F), 64, -Fabsmax, Fabsmax)
figure(2);
showgrey(imag(F), 64, -Fabsmax, Fabsmax)
figure(3);
showgrey(abs(F), 64, -Fabsmax, Fabsmax)
figure(4);
showgrey(angle(F), 64, -pi, pi);
