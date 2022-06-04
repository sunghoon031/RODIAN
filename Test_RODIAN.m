clear all; close all; clc;


%% Generate simulation data

n = 100; 
inlier_noise_sigma = 4;   
outlier_ratio = 0.8;
ub = 100; % Upper bound
mu = rand(1)*ub; % True mean of inliers

n_outliers = round(n*outlier_ratio);
n_inliers = n - n_outliers;

inliers = nan(1, n_inliers);
for i = 1:n_inliers
    inlier = normrnd(mu, inlier_noise_sigma);
    while (inlier > ub || inlier < 0)
        inlier = normrnd(mu, inlier_noise_sigma);
    end
    inliers(i) = inlier;

end
outliers = rand(1, n_outliers)*ub;
data = [inliers, outliers];

%% Precompute bin data

nBinsCombination = [2 3 4 5 7 9 11 14 17 20];
[binEdges, binTab] = CreateBinData(nBinsCombination);


%% Evaluate

rodian = RODIAN(data, binEdges, binTab, nBinsCombination);
error = abs(mu - rodian);

disp(['mu = ' num2str(mu) ', RODIAN = ', num2str(rodian) ', error = ' num2str(error)])







%%
function [binEdges, binTab] = CreateBinData(nBinsCombinations)

    % binTab tells you which bin is between which edges:
    % nBins = 2: A-----B-----C
    % nBins = 3: A---D---E---C
    % nBins = 4: A--F--B--G--C
    %
    % Then, 
    % binEdges = [A, F, D, B, E, G, C];
    % binTab = 
    %           | A-F | F-D | D-B | B-E | E-G | G-C |
    % nBins = 1 |  1  |  1  |  1  |  1  |  1  |  1  |
    % nBins = 2 |  1  |  1  |  1  |  2  |  2  |  2  |
    % nBins = 3 |  1  |  1  |  2  |  2  |  3  |  3  |
    % nBins = 4 |  1  |  2  |  2  |  3  |  3  |  4  |
    
    binEdges = [];

    for i = nBinsCombinations
        edges = linspace(0,1,i+1);
        for edge = edges
            if (~ismember(edge, binEdges))
                binEdges(end+1) = edge;
            end
        end
    end

    binEdges = sort(binEdges);
    nBinEdges = length(binEdges);

    binCenters = (binEdges(1:nBinEdges-1) + binEdges(2:nBinEdges))/2;
    nBinCenters = length(binCenters);

    binTab = zeros(length(nBinsCombinations), nBinCenters);

    for i = 1:nBinCenters
        % Column is fixed at i

        binCenter = binCenters(i);

        for j = 1:length(nBinsCombinations)
            
            % Row is fixed at j

            nBins = nBinsCombinations(j);
            
            edges = linspace(0,1,nBins+1);

            for k = 1:nBins
                upperEdge = edges(k+1);
 
                if (binCenter < upperEdge)
                    binTab(j, i) = k; 
                    break;
                end
            end
        end
    end
 
end

function est = RODIAN(data, binEdges, binTab, nBinsCombinations)
        
    n = length(data);
    
    nBinsSize = length(nBinsCombinations);
    
    %% Sort and noramlize the data
    
    data = sort(data);
    minData = data(1);
    maxData = data(end);
    data = data - minData;
    data = data/(maxData-minData);

    %% Build histogram data
    
    nBinsMax = nBinsCombinations(end);
    histogramData = zeros(nBinsSize, nBinsMax);
    % example:
    %           | cell 1 | cell 2 | cell 3 | cell 4 | 
    % nBins = 1 |   A    |   0    |   0    |   0    | 
    % nBins = 2 |   B    |   C    |   0    |   0    |
    % nBins = 3 |   D    |   E    |   F    |   0    |
    % nBins = 4 |   G    |   H    |   I    |   J    |

    j_prev = 1;
    for i = 1:n
        for j = j_prev:length(binEdges)-1
            upperEdge = binEdges(j+1);
            if (data(i) <= upperEdge)
                j_prev = j;
                binIDs = binTab(:, j);
                for k = 1:length(binIDs)
                    histogramData(k, binIDs(k)) = histogramData(k, binIDs(k)) + 1;
                end
                break;
            end
        end
    end


    %% Estimate
    
    % For each row (nBins) of histogramData, check for a repeated maximum.
    % If so, skip that row (nBins).

    [maxFreqs,maxFreqBinIDs] = max(histogramData,[],2);

    histogramData_check = histogramData - maxFreqs;
    histogramData_check = histogramData_check==0;
    histogramData_row2skip = sum(histogramData_check, 2);
    histogramData_row2skip = histogramData_row2skip > 1;

    if (ismember(1, nBinsCombinations))
        histogramData_row2skip(1) = 1;
    end
 
    
    if (min(histogramData_row2skip) > 0)
        % All histograms have equal frequencies everywhere.
        if (rem(n,2)==0)
            est = (data(n/2)+data(n/2+1))*0.5;
        else
            est = data(ceil(n/2));
        end
    else
        p_min = inf;
        ps = zeros(1, nBinsSize);

        for i = 1:nBinsSize
            if (histogramData_row2skip(i))
                continue;
            end

            nBins = nBinsCombinations(i);
            
            p = binopdf(maxFreqs(i), n, 1/nBins);
            ps(i) = p;

            if (p < p_min)
                p_min = p;
                p_min_nBins = nBins;
                p_min_cellID = maxFreqBinIDs(i);
            end
        end

        p_min_edges = linspace(0, 1, p_min_nBins+1);
        p_min_edges = p_min_edges(p_min_cellID:p_min_cellID+1);

        data(data < p_min_edges(1) | data > p_min_edges(2)) = [];

        n = length(data);
        if (rem(n,2)==0)
            est = (data(n/2)+data(n/2+1))*0.5;
        else
            est = data(ceil(n/2));
        end
    end

    %Unnormalize
    est = est*(maxData-minData) + minData;

end