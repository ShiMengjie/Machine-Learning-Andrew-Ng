function movieList = loadMovieList()
%% 函数功能：读取电影列表，也就是从txt文件中读取数据，和前面邮件分类类似
fid = fopen('movie_ids.txt');

n = 1682;  % Total number of movies 

movieList = cell(n, 1);
for i = 1:n
    line = fgets(fid);
    [~, movieName] = strtok(line, ' ');
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
