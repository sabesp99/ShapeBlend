function export_corf_gpu_geometry(output_file, sigma, beta)


if nargin < 1 || isempty(output_file), output_file = './corf_geometry.mat'; end
if nargin < 2 || isempty(sigma), sigma = 4.4; end
if nargin < 3 || isempty(beta), beta = 8; end
CORF = configureSimpleCell_fast(sigma, 0.5, 0.5);
CORF.simpleCell.inhibition = modifyModel_fast( ...
    CORF.simpleCell.excitation, 'invertpolarity', 1, 'overlappixels', beta);
excitation = CORF.simpleCell.excitation;
inhibition = CORF.simpleCell.inhibition;
radius = CORF.params.radius;
sigmaRatio = CORF.params.sigmaRatio;
sigma0 = CORF.params.sigma0;
alpha = CORF.params.alpha;
save(output_file, 'excitation', 'inhibition', 'sigma', 'beta', ...
    'radius', 'sigmaRatio', 'sigma0', 'alpha', '-v7');
fprintf('Saved geometry: %s (exc=%d, inh=%d, radius=%d)\n', ...
    output_file, size(excitation,2), size(inhibition,2), radius);
end

function filterModel = configureSimpleCell_fast(sigma, sigmaRatio, t2)
    stimulus = zeros(200);
    stimulus(:, 1:100) = 1;
    params.eta = 1;
    params.radius = ceil(sigma*2.5)*2+1;

    DoG(:,:,1) = getDoG_fast(stimulus, sigma, 0, sigmaRatio, 1, 0);
    DoG(:,:,2) = -DoG(:,:,1);
    DoG(:,:,1) = DoG(:,:,1) .* (DoG(:,:,1) > 0);
    DoG(:,:,2) = DoG(:,:,2) .* (DoG(:,:,2) > 0);

    if sigma >= 1 && sigma <= 2
        params.rho = [14.38 6.9796 3.0310 1.4135];
    elseif sigma > 2 && sigma < 2.4
        params.rho = [19.18 9.369 4.5128 2.1325];
    elseif sigma >= 2.4 && sigma <= 3.5
        params.rho = [24.62 12.6488 6.1992 3.0515];
    elseif sigma >= 4 && sigma <= 5
        params.rho = [34.43 18.08 9.2467 4.7877 3.3021];
    else
        error('Unsupported sigma=%g in original CORF configuration', sigma);
    end
    params.alpha = 0.9;

    fp = [100,100];
    simpleCell.excitation = [];
    for r = 1:length(params.rho)
        [onoff, rho, phi] = determineProperties_fast( ...
            DoG, params.rho(r), params.eta, fp, t2);
        if ~isempty(onoff)
            simpleCell.excitation = [simpleCell.excitation, ...
                [onoff; repmat(sigma,1,length(onoff)); rho; phi]]; %#ok<AGROW>
        end
    end
    simpleCell.excitation(4,:) = mod(simpleCell.excitation(4,:),2*pi);
    params.sigma0 = 2;
    params.sigmaRatio = sigmaRatio;
    filterModel.params = params;
    filterModel.params.radius = round( ...
        max(params.rho) + (2 + params.alpha*max(params.rho))/2);
    filterModel.simpleCell = simpleCell;
end

function [onoff, rho, phi] = determineProperties_fast(input, rho, eta, fp, t2)
    gb = max(input,[],3);
    t = inf;
    for i = 1:size(input,3)
        mxlist = max(max(input(:,:,i)));
        if mxlist < t
            t = mxlist;
        end
    end
    t = min(mxlist); % Retained from the supplied original implementation.
    x = 1:360*2;

    if rho > 0
        y = gb(sub2ind([size(input,1),size(input,2)], ...
            round(fp(1) + rho.*cos(x.*pi/180)), ...
            round(fp(2) + rho.*sin(x.*pi/180))));
        y = circshift(y',270)';
        if length(unique(y)) == 1
            onoff = []; rho = []; phi = [];
            return;
        end
        y(y < 0.01) = 0;
        y = round(y*1000)/1000;
        BW = bwlabel(imregionalmax(y));
        npeaks = max(BW(:));
        peaks = zeros(1,npeaks);
        for i = 1:npeaks
            peaks(i) = mean(find(BW == i));
        end
        f = peaks >= 180 & peaks < 540;
        phivalues = peaks(f);


        if phivalues(1)+360 - phivalues(end) < eta
            if P(locs(uidx(1))) < P(locs(uidx(end))) %#ok<UNRCH>
                phivalues(1) = [];
            else
                phivalues(end) = [];
            end
        end
        [x, y] = pol2cart((phivalues-270)*pi/180,rho);
        phi = phivalues*pi/180;
        onoff = zeros(1,numel(phi));
        for i = 1:numel(phi)
            [~, idx] = max(input(round(fp(2)+x(i)),round(fp(1)+y(i)),:));
            onoff(i) = idx-1;
        end
    else
        centreResponses = reshape(input(round(fp(2)),round(fp(1)),:),1,2);
        mx = max(centreResponses,[],1);
        if max(mx) >= t
            onoff = find(mx > t2*max(mx)) - 1;
            phi = zeros(1,length(onoff));
        else
            onoff = [];
            phi = [];
        end
    end
    rho = repmat(rho,1,length(phi));
end

function [output, sigma] = getDoG_fast(img, sigma, onoff, sigmaRatio, config, width, threshold) %#ok<INUSD>
    % Same FFT equation as original; reuse the Gaussian kernel FFT.
    sz = size(img) + width + width;
    key = sprintf('%d_%d_%.17g_%.17g_%d', sz(1), sz(2), sigma, sigmaRatio, onoff);
    persistent lastKey lastFftKernel
    if isempty(lastKey) || ~strcmp(lastKey, key)
        g1 = fspecial('gaussian', sz, sigma);
        g2 = fspecial('gaussian', sz, sigma*sigmaRatio);
        if onoff == 1
            G = g2 - g1;
        else
            G = g1 - g2;
        end
        lastFftKernel = fft2(G,sz(1),sz(2));
        lastKey = key;
    end
    resultimg = padarray(img,[width width],'both','symmetric');
    output = fftshift(ifft2(lastFftKernel .* fft2(resultimg)));
    if nargin == 7
        output(output < threshold*max(output(:))) = 0;
    end
end

function response = getSimpleCellResponse_fast(img, model, orientlist, inhibitionFactor, blurMode)
    [SimpleCellHashTable, weightsigma] = computeBlurredResponse_fast(img, model, blurMode);
    response = zeros(size(img,1), size(img,2), numel(orientlist));
    for i = 1:numel(orientlist)
        rotmodel.simpleCell.excitation = modifyModel_fast( ...
            model.simpleCell.excitation,'thetaoffset',orientlist(i)*pi/180);
        if isfield(model.simpleCell,'inhibition')
            rotmodel.simpleCell.inhibition = modifyModel_fast( ...
                model.simpleCell.inhibition,'thetaoffset',orientlist(i)*pi/180);
        end
        excitationresponse = getResponse_fast( ...
            SimpleCellHashTable,rotmodel.simpleCell.excitation,weightsigma,model.params);
        if isfield(model.simpleCell,'inhibition')
            inhibitionresponse = getResponse_fast( ...
                SimpleCellHashTable,rotmodel.simpleCell.inhibition,weightsigma,model.params);
            rotresponse = excitationresponse - inhibitionFactor*inhibitionresponse;
            rotresponse = rotresponse .* (rotresponse > 0);
        else
            rotresponse = excitationresponse;
        end
        response(:,:,i) = rotresponse;
    end
end

function response = getResponse_fast(SimpleCellHashTable, simpleCell, weightsigma, params)
    ntuples = size(simpleCell,2);
    w = zeros(1,ntuples);
    response = 1;
    for i = 1:ntuples
        delta = simpleCell(1,i);
        sigma = simpleCell(2,i);
        rho = round(simpleCell(3,i)*1000)/1000;
        phi = simpleCell(4,i);
        [col,row] = pol2cart(phi,rho);
        blurredResponse = SimpleCellHashTable(getHashKey([delta sigma rho]));
        shiftedResponse = imshift(blurredResponse,fix(row),-fix(col));
        w(i) = exp(-rho^2/(2*weightsigma*weightsigma));
        response = response .* shiftedResponse;
    end
    response = response.^(1/sum(w));
    response = response(params.radius+1:end-params.radius, ...
        params.radius+1:end-params.radius);
end

function [SimpleCellHashTable, weightsigma] = computeBlurredResponse_fast(img,model,blurMode)
    sigmaList = model.simpleCell.excitation(2,:);
    paramsList = model.simpleCell.excitation([1 2 3],:)';
    if isfield(model.simpleCell,'inhibition')
        sigmaList = [sigmaList model.simpleCell.inhibition(2,:)];
        paramsList = [paramsList; model.simpleCell.inhibition([1 2 3],:)'];
    end
    sigmaList = unique(sigmaList);
    paramsList = unique(round(paramsList*1000)/1000,'rows');
    nsigmaList = numel(sigmaList);
    nparamsList = size(paramsList,1);

    LGNKeyList = cell(1,nsigmaList*2);
    LGNValueList = cell(1,nsigmaList*2);
    idx = 1;
    for s = 1:nsigmaList
        delta = 0;
        LGNValueList{idx} = getDoG_fast( ...
            img,sigmaList(s),0,model.params.sigmaRatio,delta,model.params.radius);
        LGNKeyList{idx} = getHashKey([delta sigmaList(s)]);
        delta = 1;
        LGNValueList{idx+1} = -LGNValueList{idx};
        LGNKeyList{idx+1} = getHashKey([delta sigmaList(s)]);
        LGNValueList{idx} = LGNValueList{idx} .* (LGNValueList{idx} > 0);
        LGNValueList{idx+1} = LGNValueList{idx+1} .* (LGNValueList{idx+1} > 0);
        idx = idx + 2;
    end
    LGNHashTable = containers.Map(LGNKeyList,LGNValueList);
    SimpleCellKeyList = cell(1,nparamsList);
    SimpleCellValueList = cell(1,nparamsList);
    weightsigma = max(paramsList(:,3)) / 3;
    for p = 1:nparamsList
        delta = paramsList(p,1);
        sigma = paramsList(p,2);
        rho = paramsList(p,3);
        SimpleCellKeyList{p} = getHashKey([delta sigma rho]);
        LGN = LGNHashTable(getHashKey([delta sigma]));
        r = round((model.params.sigma0 + model.params.alpha*rho)/2);
        if r > 0
            if strcmp(model.params.blurringType,'Sum')
                smoothfilter = fspecial('gaussian',[2*r+1,2*r+1],r/3);
                if strcmp(blurMode, 'exact')
                    SimpleCellValueList{p} = conv2(LGN,smoothfilter,'same');
                else
                    % Mathematically equivalent Gaussian separability; near-
                    % identical floating-point results, not bit-identical.
                    mid = r+1;
                    kernelColumn = smoothfilter(:,mid);
                    kernelRow = smoothfilter(mid,:)/smoothfilter(mid,mid);
                    SimpleCellValueList{p} = conv2(kernelColumn,kernelRow,LGN,'same');
                end
            elseif strcmp(model.params.blurringType,'Max')
                SimpleCellValueList{p} = maxgaussianfilter( ...
                    LGN,r/3,0,0,[1 1],size(LGN));
            end
        else
            SimpleCellValueList{p} = LGN;
        end
        SimpleCellValueList{p} = SimpleCellValueList{p} ...
            .^ exp(-rho^2/(2*weightsigma*weightsigma));
    end
    SimpleCellHashTable = containers.Map(SimpleCellKeyList,SimpleCellValueList);
end

function model = modifyModel_fast(model,varargin)
    nargs = numel(varargin);
    names = varargin(1:2:nargs);
    values = varargin(2:2:nargs);
    for i = 1:numel(names)
        if strcmp(names(i),'invertpolarity')
            if values{i} == 1
                model(1,:) = 1-model(1,:);
            end
        elseif strcmp(names(i),'thetaoffset')
            model(4,:) = mod(model(4,:) + values{i},2*pi);
        elseif strcmp(names(i),'overlappixels')
            rho = model(3,:);
            phi = model(4,:);
            [x,y] = pol2cart(phi,rho);
            negx = x < 0;
            x(negx) = x(negx) - values{i};
            x(~negx) = x(~negx) + values{i};
            [phi,model(3,:)] = cart2pol(x,y);
            model(4,:) = mod(phi+2*pi,2*pi);
        end
    end
end
