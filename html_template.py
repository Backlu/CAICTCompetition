# -*- coding: utf-8 -*-

import copy
import numpy as np
import html_template as ht
import socket
import datetime as dt

WORK_DIR = r'/Users/jayhsu/work/github/CAICTCompetition'
IMG_DIR = WORK_DIR + r''

def set_imgdir(path):
    global IMG_DIR
    IMG_DIR = path


class Tag:
    def __init__(self, tagType, id='', name='', innerHtml='', properties={}, style={}):
        self.tagType=tagType
        self.properties=properties
        self.style=style
        self.properties['id']=id if id else ''
        self.properties['name']=name if name else ''
        self.innerHtml=innerHtml if innerHtml else ''
        self.tags=[]
        self.to_html = self.getHtml

    def addTag(self, tag):
        self.tags.append(tag)

    def addText(self, text):
        self.tags.append(Tag('_text', innerHtml=text))

    def getHtml(self, indent=True, indentCnt=0):
        bf=[]
        if self.tagType=='_text':
            if indent==True:
                bf.append('\t'*indentCnt)
            bf.append(self.innerHtml+'\n')
            #bf.append('\n')
        else:
            if indent==True:
                bf.append('\t'*indentCnt)
            bf.append('<')
            bf.append(self.tagType)

            for k, v in self.properties.items():
                if v:
                    bf.append(' %s="%s"' %(k, v))
            if self.style:
                style=''
                for k, v in self.style.items():
                    style+='%s:%s;' %(k,v)
                bf.append(' style="%s"' %style)
            bf.append('>')
            if self.innerHtml:
                bf.append('\n')
                if indent==True:
                    bf.append('\t'*(indentCnt+1))
                bf.append(self.innerHtml)
            if indent==True:
                bf.append('\n')

            for t in self.tags:
                bf.append(t.getHtml(indent=indent, indentCnt=indentCnt+1))

            if indent==True:
                bf.append('\t'*indentCnt)
            bf.append('</')
            bf+=self.tagType
            bf.append('>')
            if indent==True:
                bf.append('\n')
        return ''.join([str(_) for _ in bf])


class Html(Tag):
    def __init__(self, css='jCss_1', jQuery=True):

        Tag.__init__(self, 'html')
        head=Tag('head')
        body=Tag('body')
        self.addTag(head)
        self.addTag(body)
        self.head=head
        self.body=body

        jCSS_1='''
            body {    font-family: arial;   line-height:24px    }
            pre {    font-family: arial    }
            table {    border: 1px solid #666666;    border-collapse: collapse;    }
            td,th {    border: 1px solid #666666;    padding: 5;     font-size: 10pt;    text-align:center;    }
            th {    background-color: FFDDBB;  }
            a:link    {        color: #0000FF;        text-decoration: none;    }
            a:visited    {        color: #0000FF;        text-decoration: none;    }
            a:hover    {        color: #FF0000;        text-decoration: underline;    }
            a:active    {        color: #FF0000;        text-decoration: underline;    }
        '''

        if css=='jCss_1':
            head.addText('<meta charset="utf-8">')
            head.addTag(Tag('style', properties={'type':'text/css'}, innerHtml=jCSS_1))
        head.addTag(Tag('META', properties={'http-equiv':'Content-Type', 'content':'text/html; charset=gb2312'}))
        # head.addTag(Tag('META', properties={'http-equiv':'Content-Type', 'content':'text/html; charset=UTF-8'}))

        if jQuery==True:
            jq=Tag('script', properties={'src':'http://code.jquery.com/jquery-1.10.1.min.js'})
            head.addTag(jq)

    def save(self, fileName, encode='big5'):
        f=open(fileName, 'w')
        f.write(self.getHtml())
        f.close

class HtmlTemplate:
    def __init__(self, title='', params={}, cards=[], texts=[], controls=[], page=''):
        self.title = title
        self.params = params
        self.cards = cards
        self.texts = texts
        self.controls = controls
        self.page = page


    def to_html(self):
        html = Html(css=False, jQuery=False)
        jquery = '''
            <script type="text/javascript" src="static/js/jquery-1.12.0.js"></script>
            <script type="text/javascript" src="static/js/jquery-ui.js"></script>

            <script type="text/javascript" src="static/js/jquery.dataTables.min.js"></script>
            <script type="text/javascript" src="static/js/multiselect/multiselect.js"></script>
            <script type="text/javascript" src="static/js/jquery.easytabs.min.js"></script>
            <script type="text/javascript" src="static/js/bokeh/bokeh-0.12.0.min.js"></script>

            <script type="text/javascript">
            $( function() {
                $(".multiselect").multiselect({
                    selectedList: 1
                    , noneSelectedText : 'None'
                });

                $("table.dataframe").dataTable( {
                    "paging":false
                    , "info":false
                });

                $("#tab-container").easytabs({
                    animate: false
                });

            } );
            </script>
        '''

        # .plotdiv.table
        css = '''
          <link rel="stylesheet" href="static/js/jquery-ui.css">
          <link rel="stylesheet" href="static/js/jquery.dataTables.css">
          <link rel="stylesheet" href="static/js/multiselect/multiselect.css">
          <link rel="stylesheet" href="static/js/bokeh/bokeh-0.12.0.min.css">

		<style type="text/css">
                body {    font-family: arial;   line-height:24px    }
                pre {    font-family: arial    }
                table {    border: 1px solid #666666;    border-collapse: collapse;    }
                td,th {    border: 1px solid #666666;    padding: 5;     font-size: 12pt;    text-align:center;    }
                td { white-space: nowrap; text-align:left }
                td.ctrl { white-space: nowrap; text-align:left; background: #DDEEFF}

                table.nb {    border: 0px none #666666;    border-collapse: collapse;    }
                td.nb {    border: 0px none #666666;    padding: 5;     font-size: 12pt;    text-align:center;    }
                a:link    {        color: #0000FF;        text-decoration: none;    }
                a:visited    {        color: #0000FF;        text-decoration: none;    }
                a:hover    {        color: #FF0000;        text-decoration: underline;    }
                a:active    {        color: #FF0000;        text-decoration: underline;    }
                .plotdiv table {border: 0px solid #666666;}
                .plotdiv td {border: 0px solid #666666;}

                .etabs { margin: 0; padding: 0; }
                .tab { display: inline-block; zoom:1; *display:inline; background: #eee; border: solid 1px #999; border-bottom: none; -moz-border-radius: 4px 4px 0 0; -webkit-border-radius: 4px 4px 0 0; }
                .tab a { font-size: 14px; line-height: 2em; display: block; padding: 0 10px; outline: none; }
                .tab.active { background: #fff; padding-top: 6px; position: relative; top: 1px; border-color: #666; }
                .tab-container .panel-container { background: #fff; border: solid #666 1px; padding: 10px; -moz-border-radius: 0 4px 4px 4px; -webkit-border-radius: 0 4px 4px 4px; }
                .panel-container { margin-bottom: 10px; }

                table thead th{background-color:#e0e0ff}

		</style>
        '''

        html.body.addText(css)
        html.body.addText(jquery)

        html.body.addText('<b><font size="6">%s</font></b><br />'%self.title)

        # --- ps
        if self.params.__len__()>0:
            html.body.addText('<font color="grey" size="1">ps = %s</font>'%str({k:v for k, v in self.params.items()}))
        html.body.addText('<br />'.join(self.texts))

        # --- contorls
        if self.controls.__len__()>0:
            controls = self.controls
            html_ctrls = ('').join([_.get_html() for _ in controls])
            if self.page != '':
                html_ctrls += '''<input type="submit" onclick="go()" />'''
            html_ctrls = '<table width="100%"><tr><td class="ctrl">' + html_ctrls + '</td></tr></table>'
            html.body.addText(html_ctrls)
            # jq1 = '\n'.join(["{id} = $('#{id}').get(0).value;".format(id=ctrl.id) for ctrl in controls if ctrl.id!=''])
            jq1 = '\n'.join([ctrl.get_jquery() for ctrl in controls if ctrl.id!=''])
            # jq2 = "url = ['/{page}', {lst}].join('/');".format(page=self.page, lst = ', '.join([ctrl.id for ctrl in controls if ctrl.id!='']))
            jq2 = "url = ['/{page}', {lst}].join('/');".format(page=self.page, lst = ', '.join([ctrl.id for ctrl in controls if ctrl.id!='']))

            jq1 = ''
            jq2 = "url = ['/{page}', {lst}].join('/');".format(page=self.page, lst = ', '.join([ctrl.get_jquery() for ctrl in controls if ctrl.id!='']))
            jquery = '''
              <script language="Javascript">
                  function go(){
                      %s
                      %s
                      // alert(url);
                      window.location=url;
                  }
              </script>
            '''%(jq1, jq2)
            html.body.addText(jquery)

        # --- cards
        html.body.addText(get_cards_html(self.cards))

        return html.getHtml()


# cards = [['name1', 'text', 'aa'], ['name2', 'text', 'bb']]
def get_cards_html(cards):
    html_cards = '''
        <hr />
        <div id="tab-container" class='tab-container'>
         <ul class='etabs'>
           {buttons}
         </ul>
         <div class='panel-container'>
          {panels}
         </div>
        </div>
    '''
    buttons = []
    panels = []
    # return str(cards)
    # cards = [['Test Line', [['text', 'a']]]]
    for i, card in enumerate(cards):
        name, section = card
        buttons.append('''<li class='tab'><a href="#tab-%s">%s</a></li>'''%(name.replace(' ', '_'), name))
        txt = ''
        for type_, value in section:
            if type_=='text':
                txt = txt + '<div style="position:relative;left:30;"><pre>%s</pre></div>'%value
            elif type_=='literal':
                txt = txt + value
            elif type_=='table':
                value.columns.name=''
                txt = txt + '<div style="position:relative;left:30;width:200;">%s</div>'%value.fillna('').to_html(escape=False)
            elif type_=='img':
                txt = txt + '<div style="position:relative;left:30;"><img src="%s" /></div>'%value
            elif type_=='fig':
                img = fig2web(value)
                txt = txt + '<div style="position:relative;left:30;"><img src="%s" /></div>'%img
            else:
                txt = txt + '<div style="position:relative;left:30;">%s</div>'%value
        panels.append('''
        	<div id="tab-%s">
        		%s
        	</div>'''%(name.replace(' ', '_'), txt)
        )
    s = '\n'.join(panels)

    html_cards = html_cards.format(buttons='\n'.join(buttons), panels = s)
    return html_cards


def fig2web(fig, filename='', location=r'static/img', workdir=IMG_DIR):
    import datetime as dt
    if filename=='':
        filename = '%s.png'%dt.datetime.now().strftime('%Y%m%d%H%M%S%f')
    filepath = '/'.join([workdir,location, filename])
    fig.savefig(filepath, bbox_inches='tight')
    #print('<-- %s'%filepath)
    return '/%s/%s'%(location.replace('\\', '/'), filename)

fig2img = fig2web


class Control_Literal:
    def __init__(self, text='<br />'):
        self.id = ''
        self.text = text

    def get_jquery(self):
        return ''
    def get_html(self):
        return self.text


def get_html_select(id='', name='', values=[], labels=[], selected_value='', first_label=''):
    if labels==[]:
        labels = values
    if first_label=='':
        first_label = '--- %s ---'%name
    vals = [''] + list(values)
    lbs = [first_label] + list(labels)
    ops = zip(vals, lbs)

    html_options = '\n'.join(['<option value="%s">%s</option>'%(_[0], _[1]) for _ in ops])
    html = '<select id="{id}">{html_options}</select>'.format(id=id, html_options=html_options)
    html = html.replace('<option value="{v}">'.format(v=selected_value), '<option value="{v}" selected>'.format(v=selected_value))
    # print(html)
    return html


class Control_Select:
    def __init__(self, id, text, options=[], value='', labels=[]):
        if labels==[]:
            labels = options
        self.id = id
        self.text = text
        self.options = options
        self.value = value
        self.labels = labels

    def get_jquery(self):
        return "$('#{id}').val().replace('/', '%2F')".format(id=self.id)

    def get_html(self):
        if self.text=='':
            return '{html}'.format(
                html = ht.get_html_select(self.id, self.text, self.options, self.labels, selected_value=self.value)
            )  + '&nbsp;'*4
        else:
            return '{text}: {html}'.format(
                text=self.text
                , html = ht.get_html_select(self.id, self.text, self.options, self.labels, selected_value=self.value)
            )  + '&nbsp;'*4

class Control_Text:
    def __init__(self, id, text, value='', size=30, disabled=True):
        self.id = id
        self.text = text
        self.value = value
        self.size = size
        self.disabled = disabled
    def get_jquery(self):
        return "$('#{id}').val().replace('/', '%2F')".format(id=self.id)
        # return "{id} = $('#{id}').get(0).value;".format(id=self.id)

    def get_html(self):
        if self.disabled:
            return '{text}: <input id="{id}" size="{size}", value="{value}" disabled>'.format(id=self.id, text=self.text, size=self.size, value=self.value) + '&nbsp;'*4
        else:
            return '{text}: <input id="{id}" size="{size}", value="{value}">'.format(id=self.id, text=self.text, size=self.size, value=self.value) + '&nbsp;'*4


class Control_Textarea:
    def __init__(self, id, text, value='', size=30, disabled=False, newline=','):
        self.id = id
        self.text = text
        self.value = value
        self.size = size
        self.disabled = disabled
        self.newline = newline

    def get_jquery(self):
        # return "$('#{id}').val().replace('/', '%2F').replace('/\n/g', '{newline}')".format(id=self.id, newline=self.newline)
        return r"$('#{id}').val().replace('/', '%2F').replace(/\n/g, '{newline}')".format(id=self.id, newline=self.newline)
        # return r'''alert($('#{id}').val().replace(/\n/g, 'qq') )'''.format(id=self.id)

    def get_html(self):
        if self.disabled:
            return '{text}: <textarea id="{id}" size="{size}", value="{value}" disabled></textarea>'.format(id=self.id, text=self.text, size=self.size, value=self.value) + '&nbsp;'*4
        else:
            return '{text}: <textarea id="{id}" size="{size}", value="{value}"></textarea>'.format(id=self.id, text=self.text, size=self.size, value=self.value) + '&nbsp;'*4


class Control_Hidden:
    def __init__(self, id, value=''):
        self.id = id
        self.value = value
    def get_jquery(self):
        return "'%s'"%self.value
        # return "{id} = $('#{id}').get(0).value;".format(id=self.id)
    def get_html(self):
        return ''


def get_html_multiselect(id='', name='', values=[], labels=[], selected_values=[]):
    if labels==[]:
        labels = values
    vals = list(values)
    lbs = list(labels)
    ops = zip(vals, lbs)

    html_options = '\n'.join(['<option value="%s">%s</option>'%(_[0].replace('/', '%2F'), _[1]) for _ in ops])
    html = '<select id="{id}" multiple="multiple" class="multiselect">{html_options}</select>'.format(id=id, html_options=html_options)
    for v in selected_values:
        html = html.replace('<option value="{v}">'.format(v=v), '<option value="{v}" selected>'.format(v=v))
    # print(html)
    return html


class Control_MultiSelect:
    def __init__(self, id, text, options=[], value='', sep=','):
        self.id = id
        self.text = text
        self.options = options
        self.value = value
        self.sep = sep

    def get_jquery(self):
        # return "$('#{id}').val().join('{sep}').replace('/', '%2F')".format(id=self.id, sep=self.sep)
        return "$('#{id}').val()".format(id=self.id, sep=self.sep)

    def get_html(self):
        return '{text}: {html}'.format(
            text=self.text
            , html = ht.get_html_multiselect(self.id, self.text, self.options, selected_values=self.value.split(self.sep))
        )  + '&nbsp;'*4


class Control_DatePicker:
    def __init__(self, id, text, value='', minDate='', maxDate=''):
        self.id = id
        self.text = text
        self.value = value
        self.minDate=minDate
        self.maxDate=maxDate
    def get_jquery(self):
        return "$('#{id}').val().replace('/', '%2F')".format(id=self.id)
        # return "{id} = $('#{id}').get(0).value;".format(id=self.id)
    def get_html(self):
        ctrl = '''
        <input id="{id}" class="datepicker" size="8">
        <script>
            $( function() {{
                $("#{id}").datepicker();
                $("#{id}").datepicker("option", "dateFormat", "yy-mm-dd");
                $("#{id}").datepicker('setDate', '{value}');
                $("#{id}").datepicker('option', 'minDate', '{minDate}');
                $("#{id}").datepicker('option', 'maxDate', '{maxDate}');
            }} );
        </script>
        '''.format(id=self.id, value=self.value, minDate=self.minDate, maxDate=self.maxDate)
        return '{text}: {ctrl}'.format(text=self.text, ctrl=ctrl) + '&nbsp;'*4


class Control_HourPicker:
    def __init__(self, id, text, value='', minDate='', maxDate=''):
        self.id = id
        self.text = text
        self.value = value
        if self.value=='':
            self.value = dt.datetime.now().strftime('%Y-%m-%d_00')
        self.minDate=minDate
        self.maxDate=maxDate

    def get_jquery(self):
        return ''' $('#{id}').val() + "_" + $('#{id}_hour').val() '''.format(id=self.id)

    def get_html(self):
        date = self.value.split('_')[0]
        try:
            hour = self.value.split('_')[1]
        except:
            hour = ''
        lst = [str(_).zfill(2)+':00' for _ in range(24)]
        opts = '\n'.join([('<option>%s</option>'%_ if _!=hour else '<option selected>%s</option>'%_)  for _ in lst])
        hours = '''
            <select id="{id}_hour" style="width: 70px;" >
            {opts}
            </select>
        '''.format(id=self.id, opts=opts)
        ctrl = '''
        <input id="{id}" class="datepicker" size="8"> :{hours}
        <script>
            $( function() {{
                $("#{id}").datepicker();
                $("#{id}").datepicker("option", "dateFormat", "yy-mm-dd");
                $("#{id}").datepicker('setDate', '{value}');
                $("#{id}").datepicker('option', 'minDate', '{minDate}');
                $("#{id}").datepicker('option', 'maxDate', '{maxDate}');
            }} );
        </script>
        '''.format(id=self.id, value=date, hours=hours, minDate=self.minDate, maxDate=self.maxDate)
        return '{text}: {ctrl}'.format(text=self.text, ctrl=ctrl) + '&nbsp;'*4
