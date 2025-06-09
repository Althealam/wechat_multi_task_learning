target = ["read_comment", 
          "like", 
          "click_avatar", 
          "forward"]

sparse_features = ['userid', 
                   'feedid', 
                   'authorid', 
                   'bgm_song_id', 
                   'bgm_singer_id']

varlen_features = ['manual_tag_list',
                   'manual_keyword_list', 'description']

dense_features = ['videoplayseconds']

feature_names = sparse_features+varlen_features+dense_features
