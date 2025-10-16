# file: app/forms.py
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SubmitField
from wtforms.validators import DataRequired

# Buat form input sesuai fitur pada dataset
class CreditForm(FlaskForm):
    LIMIT_BAL = FloatField('Limit Balance (Jumlah Kredit)', validators=[DataRequired()])
    SEX = IntegerField('Jenis Kelamin (1 = Pria, 2 = Wanita)', validators=[DataRequired()])
    EDUCATION = IntegerField('Tingkat Pendidikan (1=graduate school,2=university,3=high school,4=others)', validators=[DataRequired()])
    MARRIAGE = IntegerField('Status Pernikahan (1=married,2=single,3=others)', validators=[DataRequired()])
    AGE = IntegerField('Usia', validators=[DataRequired()])
    PAY_0 = IntegerField('Status pembayaran September', validators=[DataRequired()])
    PAY_2 = IntegerField('Status pembayaran Agustus', validators=[DataRequired()])
    PAY_3 = IntegerField('Status pembayaran Juli', validators=[DataRequired()])
    PAY_4 = IntegerField('Status pembayaran Juni', validators=[DataRequired()])
    PAY_5 = IntegerField('Status pembayaran Mei', validators=[DataRequired()])
    PAY_6 = IntegerField('Status pembayaran April', validators=[DataRequired()])
    BILL_AMT1 = FloatField('Tagihan September', validators=[DataRequired()])
    BILL_AMT2 = FloatField('Tagihan Agustus', validators=[DataRequired()])
    BILL_AMT3 = FloatField('Tagihan Juli', validators=[DataRequired()])
    BILL_AMT4 = FloatField('Tagihan Juni', validators=[DataRequired()])
    BILL_AMT5 = FloatField('Tagihan Mei', validators=[DataRequired()])
    BILL_AMT6 = FloatField('Tagihan April', validators=[DataRequired()])
    PAY_AMT1 = FloatField('Pembayaran September', validators=[DataRequired()])
    PAY_AMT2 = FloatField('Pembayaran Agustus', validators=[DataRequired()])
    PAY_AMT3 = FloatField('Pembayaran Juli', validators=[DataRequired()])
    PAY_AMT4 = FloatField('Pembayaran Juni', validators=[DataRequired()])
    PAY_AMT5 = FloatField('Pembayaran Mei', validators=[DataRequired()])
    PAY_AMT6 = FloatField('Pembayaran April', validators=[DataRequired()])

    submit = SubmitField('Prediksi')
