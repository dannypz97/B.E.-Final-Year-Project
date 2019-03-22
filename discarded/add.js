$(document).ready(function(){
  $('.tag-list').on("keypress", function(e) {
          if (e.keyCode == 13) {
            e.preventDefault();

            var tag =  $('#tag-list').val();

            tag = tag.replace(/[^a-zA-Z0-9\s()]+/g, ''); //remove chars that are not alphanumeric, space or ( )
            tag = tag.replace(/'  '/g,' '); //replace multiple spaces with single space
            $('.chosen-tags').append('<a href="#" class="badge badge-primary">' + tag + '</a>');

            $('#tag-list').val("");

          }
  });

  $('body').on('click', '.badge-primary', function () {
       $(this).remove();
  });

  $('.btn-add').on('click',function(e){
    e.preventDefault();
    $('.question-blocks').append($('.question-block').html());
  });
});
