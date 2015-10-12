/*
 * A Wolfram Share Widget Plugin that outputs share widget sidebar
 * @author Team Nick
 * @example try {
 *      $.addWolframShareWidget();
 * } catch(e) {
 *
 * }
 * 
 * @version 1.0.3
 * @desc This depends on share.aside.css file
 */

 "use strict";

;(function (window, $){
    /**
     * @param {Object<String>=} options Give additional options for the widget
     * @param {String} options.topValue set the container value
     * @param {String} options.fontSizeValue set the font size for the sidebar
     */
    $.addWolframShareWidget = function(options){
        
	var defaults = {
            topValue: "250px",
            fontSizeValue: "11px"
        }
	
        // if not or partially defined, options will have default value
        options = $.extend({}, defaults, options);
        var hostName = {};
	
        switch (window.location.hostname) {
            case 'www.devel.wolframalpha.com' :
                hostName = 'www.devel.wolframalpha.com';
                break;
            case 'www.current.wolframalpha.com' :
                hostName = 'www.current.wolframalpha.com';
                break;
            case 'www.test.wolframalpha.com' :
                hostName = 'www.test.wolframalpha.com';
                break;
            case 'preview.wolframalpha.com' :
                hostName = 'preview.wolframalpha.com';
                break;
            case 'www.wolframalpha.com' :
                hostName = 'www.wolframalpha.com';
                break;
        }
        
        var shortURL =  encodeURIComponent(window.location.href);
        var QueryString = function () {
          // This function is anonymous, is executed immediately and 
          // the return value is assigned to QueryString!
          var query_string = {};
          var query = window.location.search.substring(1);
          var vars = query.split("&");
          for (var i=0;i<vars.length;i++) {
            var pair = vars[i].split("=");
              // If first entry with this name
            if (typeof query_string[pair[0]] === "undefined") {
              query_string[pair[0]] = pair[1];
              // If second entry with this name
            } else if (typeof query_string[pair[0]] === "string") {
              var arr = [ query_string[pair[0]], pair[1] ];
              query_string[pair[0]] = arr;
              // If third or later entry with this name
            } else {
              query_string[pair[0]].push(pair[1]);
            }
          } 
            return query_string;
        } ();
        
        var pubKey="1c0e0d54-547d-4312-9dde-61539c24e22f";
        var accessKey="862353f2d9a789be97ae70fc110aca45";
	
        var title = "WolframAlpha - " + encodeURIComponent(QueryString.i);
        var tumblrTitle = "WolframAlpha - " + encodeURIComponent(encodeURIComponent(QueryString.i));
        var emailMsg= "%0D%0A%0D%0A%0D%0ACheck this out on Wolfram%7cAlpha.%0D%0A%0D%0AOn the web:%0D%0A" + shortURL + "&mail=1%0D%0A%0D%0A";
        var twitterURL="http://wd.sharethis.com/api/sharer.php?title=" + title +
                      "&amp;pub_key=" + pubKey +
                      "&amp;access_key=" + accessKey +
                      "&amp;via=wolfram_alpha" +
                      "&amp;shorten=false" +
                      "&amp;url=" + shortURL +
                      "&amp;destination=twitter";
        var fbURL =   "http://wd.sharethis.com/api/sharer.php?title=" + title +
                      "&amp;pub_key=" + pubKey +
                      "&amp;url=" + shortURL +
                      "&amp;destination=facebook" +
                      "&amp;access_key=" + accessKey;
        var tumblrURL="http://wd.sharethis.com/api/sharer.php?title=" + tumblrTitle +
        			  "&amp;pub_key=" + pubKey +
        			  "&amp;access_key=" + accessKey +
        			  "&amp;url=" + shortURL +
        			  "&amp;destination=tumblr";
        var otherURL="http://wd.sharethis.com/api/sharer.php?title=" + title +
                      "&amp;pub_key=" + pubKey +
                      "&amp;access_key=" + accessKey +
                      "&amp;url=" + shortURL +
                      "&amp;destination=";

        var content =  '<img id="share-txt" src="//www.wolframcdn.com/images/sharepage.png" />' + 
                       '<ul>' +
                           '<li id="fbcontain">' + 
                              '<a href="' + fbURL + '"' + 'class="facebook jump fb on" target="_blank" title="Share via Facebook">Facebook</a>' +
                               '<div id="fb-root" class="fbad">' +
                                   //'<script async="" src="http://connect.facebook.net/en_US/sdk.js"></script>' +
                                   '<div style="position: absolute; top: -10000px; height: 0px; width: 0px;">' +
                                   // '<div style="position: absolute; top: -10000px; height: 0px; width: 0px;"></div></div>' +
                                     '<div>' +
                                       '<iframe name="fb_xdm_frame_http" frameborder="0" allowtransparency="true" scrolling="no" id="fb_xdm_frame_http" aria-hidden="true" title="Facebook Cross Domain Communication Frame" tab-index="-1" src="http://static.ak.facebook.com/connect/xd_arbiter/wTH8U0osOYl.js?version=40#channel=f25b16b08c&amp;origin=http%3A%2F%2F' + hostName + '" style="border: none;"></iframe>' +
                                       '<iframe name="fb_xdm_frame_https" frameborder="0" allowtransparency="true" scrolling="no" id="fb_xdm_frame_https" aria-hidden="true" title="Facebook Cross Domain Communication Frame" tab-index="-1" src="https://s-static.ak.facebook.com/connect/xd_arbiter/wTH8U0osOYl.js?version=40#channel=f25b16b08c&amp;origin=http%3A%2F%2F' + hostName + '" style="border: none;"></iframe>' +
                                     '</div>' +
                                   '</div>' +
                               '</div>' +
                               '<div id="fbad" class="callout popup">' +
                                  '<a href="' + fbURL + '"' + 'class="fb" target="_blank" title="Share via Facebook">Share via Facebook </a>' +
                                  '<a id="fbimgs" href="/facebook/">Generate a free interactive personal analytics report from your Facebook data</a>' +
                               '</div>' +
                           '</li>' +

                           '<li>' +
                              '<a href="' + twitterURL + '"' + 'class="twitter" target="_blank" title="Share via Twitter">Twitter</a>' +
                           '</li>' +

                           '<li>' +
                              '<a href="' + otherURL + 'reddit" class="reddit" target="_blank" title="Share via Reddit">Reddit</a>' +
                           '</li>' +

                           '<li>' +
                              '<a href="' + tumblrURL + '"' + 'class="tumblr" target="_blank" title="Share via Tumblr">Tumblr</a>' +
                           '</li>' +

                         '</ul>' +
                         '<ul id="moreSocial">' +
                           '<li>' +
                              '<a href="' + otherURL + 'linkedin" class="linkedin" target="_blank" title="Share via LinkedIn" style="display:none">LinkedIn</a>' +
                           '</li>' +
                           '<li>' +
                              '<a href="' + otherURL + 'digg" class="digg" target="_blank" title="Share via Digg" style="display:none">Digg</a>' +
                           '</li>' +
                           '<li>' +
                              '<a href="' + otherURL + 'stumbleupon" class="stumbleupon" target="_blank" title="Share via StumbleUpon" style="display:none">StumbleUpon</a>' +
                           '</li>' +
                           '<li>' +
                              '<a href="mailto:?subject=WolframAlpha%3a%20' + QueryString.i + '&amp;body=' + emailMsg + '" class="mail" target="_blank" title="Share via email" style="display:none">email</a>' +
                           '</li>' +
                         '</ul>' +
                         '<span id="shortUrl"><label for="shortUrlBox">Share the link directly: </label><input type="text" id="shortUrlBox" readonly="true" value="http://wolfr.am/"></span>' +
                         '<div id="share-more" class="more"><span id="more-less">More</span></div>';

	
        var shareWidgetContainer = '<aside id="share" class="sidebarContent withBannerAd"></aside>';
        
        $('#sidebar').prepend(shareWidgetContainer);

        $('#share').html(content);
	
        // toggle opening and closing on more social links box
        $('#share-more').on("click", function(e) {
            //e.preventDefault();
		
            var div = $("#share-more");  
            var aside = $("#share");
            var examples = $("#examplesContainer"); 
            if (div.hasClass("more")) {
                aside.css("height", "240px");
                div.removeClass("more"); 
                div.addClass("less");
                $('#share').addClass('target');
                $('#share .digg').attr('style', '');
                $('#share .linkedin').attr('style', '');
                $('#share .stumbleupon').attr('style', '');
                $('#share .mail').attr('style', '');
                $("#more-less").text("Less");
                if (examples.length > 0) {
                    examples.css("top", parseInt(examples.css("top").replace(/[^-\d\.]/g, '')) + 123 + "px");
                }   
            } else {
		
                aside.css("height", "107px"); 
                div.removeClass("less");
                div.addClass("more");
                $('#share').removeClass('target');
                $('#share .digg').attr('style', 'display:none');
                $('#share .linkedin').attr('style', 'display:none');
                $('#share .stumbleupon').attr('style', 'display:none');
                $('#share .mail').attr('style', 'display:none');
                $("#more-less").text("More");
                if (examples.length > 0) {
                    examples.css("top", parseInt(examples.css("top").replace(/[^-\d\.]/g, '')) - 123 + "px"); 
                }  
           }  
        });

    }
}(window, jQuery));
