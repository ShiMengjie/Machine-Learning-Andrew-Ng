function movieList = loadMovieList()
%% 读取电影列表
fid = fopen('movie_ids.txt');

n = 1682;  % Total number of movies 

movieList = cell(n, 1);
for i = 1:n
    % Read line
    line = fgets(fid);
    % Word Index (can ignore since it will be = i)
    [~, movieName] = strtok(line, ' ');
    % Actual Word
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
