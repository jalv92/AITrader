   M S B A M L   `   `   `  ����F  BAITrader.UI, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null5clr-namespace:AITrader.UI.ViewsAITrader.UI.Views  5   	   $    AITrader.UI.Views.SplashWindow   5   	   W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5   	   R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   1localclr-namespace:AITrader.UI.Views     5   	     � Title$  AITrader - Iniciando��5   	   $	��250��6&   $	��450��63    � WindowStartupLocation$ CenterScreen=�5	   	    � WindowStyle$
 None=�5
   	    � AllowsTransparency$
 True��5   	   ���#202020.��5      �� ���#2196F3$��1q�6#   $��5q�67   ._�5   
   �  � RowDefinitions 5      �� 5      $
��Auto �6    �� 5      $��* �6    �� 5      $
��Auto �6    .+�5      �� $��0��6   $��AITrader��5      $��28(�5      $
��Bold"�5      ���#2196F3$��Center=�5      $��0,20,0,0q�5      �� 5      $��1��6   $��Center=�6&   .+�5       �� $��Iniciando servicios...��6   
�������$��16(�5"      $��Center=�5#      $
��0,10q�5$      � 5&      $	��300��6   $��10��5'       � IsIndeterminate$
 True��5(      $
��0,15q�5)      
������ ���#2196F35+      �� 5-      -   5         ��Name$ StatusTextBlock��5-      $)��#Inicializando el motor de Python...��5.      ���#B0B0B0$��12(�50      $��Center=�51      $
��Wrap=�52      $
��20,5q�53      54      �� 57      $��2��6   $��© 2025 AITrader��58      ���#B0B0B0$��10(�5:      $��Center=�5;      $��0,0,0,10q�5<      5=      5>      5?      