function word_indices = processEmail(email_contents)
%% 把一封邮件的内容转换成文字特征标记

vocabList = getVocabList();

word_indices = [];

% ========================== Preprocess Email ===========================
% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);
%% 按照文档中的顺序第邮件内容进行处理
% 1.把邮件内容的字符串转换成小写格式
email_contents = lower(email_contents);

% 2.用空白符替换HTML的标记tag
% 正则表达式：替换以“<”开头和以“>”结尾，但不包含“<>”的字符串
% [ ]中，^表示非
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% 3.处理链接
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');
                       
% 4.处理邮箱地址
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
                   
% 5.处理数字
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% 6.处理金钱符号
email_contents = regexprep(email_contents, '[$]+', 'dollar');

% ========================== Tokenize Email ===========================

% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)

    % 把邮件内容进行切分，char(num)表示ASCII表中标号为num的字符
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' newline char(13)]);
   
    % 7.把非字符（space、tab等）消除
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end
    
	%[str]
    % Look up the word in the dictionary and add to word_indices if
    % found
    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to add the index of str to
    %               word_indices if it is in the vocabulary. At this point
    %               of the code, you have a stemmed word from the email in
    %               the variable str. You should look up str in the
    %               vocabulary list (vocabList). If a match exists, you
    %               should add the index of the word to the word_indices
    %               vector. Concretely, if str = 'action', then you should
    %               look up the vocabulary list to find where in vocabList
    %               'action' appears. For example, if vocabList{18} =
    %               'action', then, you should add 18 to the word_indices 
    %               vector (e.g., word_indices = [word_indices ; 18]; ).
    % 
    % Note: vocabList{idx} returns a the word with index idx in the
    %       vocabulary list.
    % 
    % Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    %       str2). It will return 1 only if the two strings are equivalent.
    %
	
	
	for i=1:length(vocabList)
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