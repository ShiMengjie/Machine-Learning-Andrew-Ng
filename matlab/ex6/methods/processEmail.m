function word_indices = processEmail(email_contents)
%% 函数功能：把文字转换成向量
% 那么问题来了，word to vector，怎么实现？
% 把文件对象中的元素和词袋中的元素依次作对比，把元素在词袋中的下标添加进文件向量中

%% 获取词袋
vocabList = getVocabList();

%% 按照文档中的顺序第邮件内容进行处理
% 1.把邮件内容的字符串转换成小写格式
email_contents = lower(email_contents);
% 2.用空白符替换HTML的标记tag
% 正则表达式：替换以“<”开头和以“>”结尾，但不包含“<>”的字符串
% [ ]中，^表示非
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
% 3.处理链接，用httpaddr代替链接内容
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');               
% 4.处理邮箱地址，用emailaddr代替邮箱地址
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
% 5.处理数字，用number代替所有数字
email_contents = regexprep(email_contents, '[0-9]+', 'number');
% 6.处理金钱符号，用dollar代替$
email_contents = regexprep(email_contents, '[$]+', 'dollar');

% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)
    % 把邮件内容进行切分，char(num)表示ASCII表中标号为num的字符
    [str, email_contents] = strtok(email_contents,[' @$/#.-:&*+=[]?!(){},''">_<;%' newline char(13)]);
    % 7.把非字符（space、tab等）消除
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; 
        continue;
    end
    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end
    
	% 文件的下标向量
	word_indices = [];
	for i=1:length(vocabList)
    % 比较两个字符差串是否相等
	match(i)=strcmp(vocabList{i}, str);
		if (match(i)==1)
			word_indices = [word_indices; i];
		else
			word_indices = [word_indices;[]];
        end
    end	
    % =============================================================
    % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;
end

end